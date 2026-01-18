import importlib
import logging
import sys
from typing import Any, Dict, List, Optional
from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def _get_model_type_from_template(tmpl: str) -> str:
    """
    Args:
        tmpl: The template kind, e.g. `regression/v1`.

    Returns:
        A model type literal compatible with the mlflow evaluation service, e.g. regressor.
    """
    if tmpl == 'regression/v1':
        return 'regressor'
    if tmpl == 'classification/v1':
        return 'classifier'
    raise MlflowException(f'No model type for template kind {tmpl}', error_code=INVALID_PARAMETER_VALUE)