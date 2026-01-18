import importlib
import logging
import sys
from typing import Any, Dict, List, Optional
from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def _get_extended_task(recipe: str, positive_class: str) -> str:
    """
    Args:
        step_config: Step config

    Returns:
        Extended type string. Currently supported types are: "regression",
        "binary_classification", "multiclass_classification"
    """
    if 'regression' in recipe:
        return 'regression'
    elif 'classification' in recipe:
        if positive_class is not None:
            return 'classification/binary'
        else:
            return 'classification/multiclass'
    raise MlflowException(f'No model type for template kind {recipe}', error_code=INVALID_PARAMETER_VALUE)