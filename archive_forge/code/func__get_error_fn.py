import importlib
import logging
import sys
from typing import Any, Dict, List, Optional
from mlflow.exceptions import BAD_REQUEST, MlflowException
from mlflow.models import EvaluationMetric, make_metric
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def _get_error_fn(tmpl: str, use_probability: bool=False, positive_class: Optional[str]=None):
    """
    Args:
        tmpl: The template kind, e.g. `regression/v1`.

    Returns:
        The error function for the provided template.
    """
    if tmpl == 'regression/v1':
        return lambda predictions, targets: predictions - targets
    if tmpl == 'classification/v1':
        if use_probability:

            def error_rate(true_label, predicted_positive_class_proba):
                if true_label == positive_class:
                    return 1 - predicted_positive_class_proba
                else:
                    return predicted_positive_class_proba
            return lambda predictions, targets: [error_rate(x, y) for x, y in zip(targets, predictions)]
        else:
            return lambda predictions, targets: predictions != targets
    raise MlflowException(f'No error function for template kind {tmpl}', error_code=INVALID_PARAMETER_VALUE)