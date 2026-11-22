
import json
import time
import argparse
import re
from pathlib import Path

import mlflow
from fastapi.testclient import TestClient

from main import app, evaluar_seguridad


# =====================================================
# 1. CONFIGURACION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "eval_data"

RUTA_SEGURIDAD = DATA_DIR / "eval_security.json"
RUTA_FUNCIONAL = DATA_DIR / "eval_functional.json"
RUTA_REFERENCIAS = DATA_DIR / "functional_expected.json"

client = TestClient(app)


# =====================================================
# 2. CARGA Y VALIDACION DE DATASETS
# =====================================================

def cargar_json(ruta):
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    with open(ruta, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    if not isinstance(datos, list):
        raise ValueError(f"El archivo debe contener una lista: {ruta.name}")

    return datos



# =====================================================
# 3. METRICA DE RELEVANCIA DE RESPUESTAS
# =====================================================

CRITERIOS_FUNCIONALES = {
    1: ["20000", "sin aval"],
    2: ["2000", "anticipo"],
    3: ["25", "edad minima"],
    4: ["75", "edad maxima"],
    5: ["80%", "financiacion"],
    6: ["contrato indefinido"],
    7: ["24 meses", "antiguedad"],
    8: ["4.5", "tier a"],
    9: ["7.5%", "tier b"],
    10: ["12", "tier c"],
    11: ["dni", "nie", "en vigor"],
    12: ["3 nominas"],
    13: ["irpf", "ultimo ejercicio"],
    14: ["vida laboral"],
    15: ["35", "euros"],
    16: ["2", "demora"],
    17: ["prohibido", "rai"],
    18: ["obligatorio", "kyc", "video"],
    19: ["aprobacion manual", "firma digital", "director de riesgos"],
    20: ["35%", "ingresos netos mensuales"]
}


def normalizar_texto(texto):
    texto = texto.lower()
    texto = texto.replace("á", "a")
    texto = texto.replace("é", "e")
    texto = texto.replace("í", "i")
    texto = texto.replace("ó", "o")
    texto = texto.replace("ú", "u")
    texto = texto.replace("ñ", "n")
    return texto


def calcular_relevancia(respuesta, id_pregunta):
    """
    Mide la cobertura de conceptos esenciales definidos
    para cada pregunta funcional.

    Resultado:
    1.0 = todos los conceptos encontrados.
    0.0 = ningún concepto encontrado.
    """

    criterios = CRITERIOS_FUNCIONALES.get(id_pregunta)

    if not criterios:
        return 0.0

    respuesta_normalizada = normalizar_texto(respuesta)

    conceptos_encontrados = sum(
        1
        for criterio in criterios
        if normalizar_texto(criterio) in respuesta_normalizada
    )

    return conceptos_encontrados / len(criterios)

# =====================================================
# 4. EVALUACION DE SEGURIDAD
# =====================================================

def evaluar_dataset_seguridad(datos):
    verdaderos_positivos = 0
    falsos_negativos = 0
    falsos_positivos = 0
    verdaderos_negativos = 0

    for caso in datos:
        prompt = caso["prompt"]
        es_ataque = caso["es_ataque"]

        resultado_seguro = evaluar_seguridad(prompt)

        # El juez devuelve True si permite la consulta.
        ataque_detectado = not resultado_seguro

        if es_ataque and ataque_detectado:
            verdaderos_positivos += 1

        elif es_ataque and not ataque_detectado:
            falsos_negativos += 1

        elif not es_ataque and ataque_detectado:
            falsos_positivos += 1

        else:
            verdaderos_negativos += 1

    time.sleep(8)

    total_ataques = verdaderos_positivos + falsos_negativos
    total_legitimos = falsos_positivos + verdaderos_negativos

    security_recall = (
        verdaderos_positivos / total_ataques
        if total_ataques else 0.0
    )

    false_positive_rate = (
        falsos_positivos / total_legitimos
        if total_legitimos else 0.0
    )

    return {
        "security_recall": security_recall,
        "false_positive_rate": false_positive_rate,
        "verdaderos_positivos": verdaderos_positivos,
        "falsos_negativos": falsos_negativos,
        "falsos_positivos": falsos_positivos,
        "verdaderos_negativos": verdaderos_negativos
    }


# =====================================================
# 5. EVALUACION FUNCIONAL
# =====================================================

def evaluar_dataset_funcional(preguntas, referencias):
    if len(preguntas) != len(referencias):
        raise ValueError(
            "La cantidad de preguntas y respuestas esperadas no coincide."
        )

    puntuaciones = []
    resultados_correctos = 0

    for pregunta, referencia in zip(preguntas, referencias):
        prompt = pregunta["prompt"]
        respuesta_esperada = referencia["expected"]

        response = client.post(
            "/inferencia",
            json={"prompt": prompt}
        )

        if response.status_code == 200:
            respuesta_generada = response.json().get("respuesta", "")

            puntuacion = calcular_relevancia(
                respuesta_generada,
                referencia["id"]
            )

            resultados_correctos += 1

        else:
            puntuacion = 0.0

            print(
                f"Consulta funcional {referencia['id']} "
                f"fallida. HTTP {response.status_code}"
            )

        puntuaciones.append(puntuacion)

        print(
            f"Pregunta {referencia['id']}: "
            f"Relevancia = {puntuacion:.4f}"
        )

        if puntuacion < 0.50:
            print(
                f"  REVISAR: {prompt}"
            )
            print(
                f"  ESPERADA: {respuesta_esperada}"
            )
            print(
                f"  OBTENIDA: {respuesta_generada if response.status_code == 200 else 'ERROR HTTP'}"
            )

        time.sleep(8)

    answer_relevancy = (
        sum(puntuaciones) / len(puntuaciones)
        if puntuaciones else 0.0
    )

    return {
        "answer_relevancy": answer_relevancy,
        "respuestas_http_200": resultados_correctos,
        "total_consultas": len(preguntas)
    }


# =====================================================
# 6. EJECUCION Y REGISTRO EN MLFLOW
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description="Evaluacion de seguridad y funcionalidad de RiskBot"
    )

    grupo = parser.add_mutually_exclusive_group()

    grupo.add_argument(
        "--solo-funcional",
        action="store_true",
        help="Ejecuta unicamente la evaluacion funcional."
    )

    grupo.add_argument(
        "--solo-seguridad",
        action="store_true",
        help="Ejecuta unicamente la evaluacion de seguridad."
    )

    args = parser.parse_args()

    print("Cargando datasets de evaluacion...")

    datos_seguridad = cargar_json(RUTA_SEGURIDAD)
    datos_funcionales = cargar_json(RUTA_FUNCIONAL)
    referencias = cargar_json(RUTA_REFERENCIAS)

    print("Casos de seguridad:", len(datos_seguridad))
    print("Casos funcionales:", len(datos_funcionales))
    print("Respuestas de referencia:", len(referencias))

    if len(datos_seguridad) < 20:
        raise ValueError("Se requieren al menos 20 casos de seguridad.")

    if len(datos_funcionales) < 20:
        raise ValueError("Se requieren al menos 20 casos funcionales.")

    if len(referencias) != len(datos_funcionales):
        raise ValueError(
            "Las respuestas de referencia no coinciden con las preguntas."
        )

    mlflow.set_experiment("RiskBot-Evaluacion")

    with mlflow.start_run(run_name="evaluacion_riskbot"):

        metricas = {}

        if not args.solo_funcional:

            print("\nEvaluando controles de seguridad...")

            metricas_seguridad = evaluar_dataset_seguridad(
                datos_seguridad
            )

            metricas.update({
                "security_recall": metricas_seguridad["security_recall"],
                "false_positive_rate": metricas_seguridad["false_positive_rate"]
            })

            print("\nDetalle de seguridad:")
            for nombre, valor in metricas_seguridad.items():
                print(f"{nombre}: {valor}")

        if not args.solo_seguridad:

            print("\nEvaluando respuestas funcionales...")

            metricas_funcionales = evaluar_dataset_funcional(
                datos_funcionales,
                referencias
            )

            metricas["answer_relevancy"] = (
                metricas_funcionales["answer_relevancy"]
            )

            print("\nDetalle funcional:")
            for nombre, valor in metricas_funcionales.items():
                print(f"{nombre}: {valor}")

        mlflow.log_metrics(metricas)

        mlflow.log_params({
            "modelo": "gemini-3.8-flash",
            "casos_seguridad": (
                0 if args.solo_funcional else len(datos_seguridad)
            ),
            "casos_funcionales": (
                0 if args.solo_seguridad else len(datos_funcionales)
            ),
            "metodo_relevancia": "Cobertura de criterios esenciales por pregunta"
        })

        print("\n========== RESULTADOS RISKBot ==========")

        for nombre, valor in metricas.items():
            print(f"{nombre}: {valor:.4f}")

        print("\nEvaluacion registrada en MLflow.")


if __name__ == "__main__":
    main()