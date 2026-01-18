import inspect
import itertools
import logging
import os
from typing import Any, Dict, Optional
import yaml
import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.validation import _is_numeric
def _get_autolog_metrics(fitted_model):
    result_metrics = {}
    failed_evaluating_metrics = set()
    for metric in _autolog_metric_allowlist:
        try:
            if hasattr(fitted_model, metric):
                metric_value = getattr(fitted_model, metric)
                if _is_numeric(metric_value):
                    result_metrics[metric] = metric_value
        except Exception:
            failed_evaluating_metrics.add(metric)
    if len(failed_evaluating_metrics) > 0:
        _logger.warning(f'Failed to autolog metrics: {', '.join(sorted(failed_evaluating_metrics))}.')
    return result_metrics