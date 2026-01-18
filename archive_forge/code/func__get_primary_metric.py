import importlib
import logging
import sys
from typing import Any, Dict, List, Optional
from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def _get_primary_metric(configured_metric: str, ext_task: str):
    if configured_metric is not None:
        return configured_metric
    else:
        return DEFAULT_METRICS[ext_task]