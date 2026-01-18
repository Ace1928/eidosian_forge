import os
import tempfile
import types
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional
import numpy as np
import yaml
import mlflow
import mlflow.utils.autologging_utils
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_package_name
from mlflow.utils.uri import append_to_uri_path
def _merge_environments(shap_environment, model_environment):
    """
    Merge conda environments of underlying model and shap.

    Args:
        shap_environment: SHAP conda environment.
        model_environment: Underlying model conda environment.
    """
    merged_conda_channels = _union_lists(shap_environment['channels'], model_environment['channels'])
    merged_conda_channels = [x for x in merged_conda_channels if x != 'conda-forge']
    shap_conda_deps, shap_pip_deps = _get_conda_and_pip_dependencies(shap_environment)
    model_conda_deps, model_pip_deps = _get_conda_and_pip_dependencies(model_environment)
    merged_conda_deps = _union_lists(shap_conda_deps, model_conda_deps)
    merged_pip_deps = _union_lists(shap_pip_deps, model_pip_deps)
    return _mlflow_conda_env(additional_conda_deps=merged_conda_deps, additional_pip_deps=merged_pip_deps, additional_conda_channels=merged_conda_channels)