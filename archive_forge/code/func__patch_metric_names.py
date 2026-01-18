import functools
import json
import logging
import os
import tempfile
from copy import deepcopy
from typing import Any, Dict, Optional
import yaml
from packaging.version import Version
import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.sklearn import _SklearnTrainingSession
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.autologging_utils import (
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.mlflow_tags import (
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_pinned_requirement
def _patch_metric_names(metric_dict):
    patched_metrics = {metric_name.replace('@', '_at_'): value for metric_name, value in metric_dict.items()}
    changed_keys = set(patched_metrics.keys()) - set(metric_dict.keys())
    if changed_keys:
        _logger.info('Identified one or more metrics with names containing the invalid character `@`. These metric names have been sanitized by replacing `@` with `_at_`, as follows: %s', ', '.join(changed_keys))
    return patched_metrics