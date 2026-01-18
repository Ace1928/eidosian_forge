import logging
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from packaging.version import Version
import mlflow
from mlflow.environment_variables import MLFLOW_ENV_ROOT
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.utils.conda import _PIP_CACHE_DIR
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.environment import (
from mlflow.utils.file_utils import remove_on_error
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd, _join_commands
from mlflow.utils.requirements_utils import _parse_requirements
def _get_python_env(local_model_path):
    """Constructs `_PythonEnv` from the model artifacts stored in `local_model_path`. If
    `python_env.yaml` is available, use it, otherwise extract model dependencies from `conda.yaml`.
    If `conda.yaml` contains conda dependencies except `python`, `pip`, `setuptools`, and, `wheel`,
    an `MlflowException` is thrown because conda dependencies cannot be installed in a virtualenv
    environment.

    Args:
        local_model_path: Local directory containing the model artifacts.

    Returns:
        `_PythonEnv` instance.

    """
    model_config = Model.load(local_model_path / MLMODEL_FILE_NAME)
    python_env_file = local_model_path / _get_python_env_file(model_config)
    conda_env_file = local_model_path / _get_conda_env_file(model_config)
    requirements_file = local_model_path / _REQUIREMENTS_FILE_NAME
    if python_env_file.exists():
        return _PythonEnv.from_yaml(python_env_file)
    else:
        _logger.info('This model is missing %s, which is because it was logged in an older versionof MLflow (< 1.26.0) that does not support restoring a model environment with virtualenv. Attempting to extract model dependencies from %s and %s instead.', _PYTHON_ENV_FILE_NAME, _REQUIREMENTS_FILE_NAME, _CONDA_ENV_FILE_NAME)
        if requirements_file.exists():
            deps = _PythonEnv.get_dependencies_from_conda_yaml(conda_env_file)
            return _PythonEnv(python=deps['python'], build_dependencies=deps['build_dependencies'], dependencies=[f'-r {_REQUIREMENTS_FILE_NAME}'])
        else:
            return _PythonEnv.from_conda_yaml(conda_env_file)