import collections
import functools
import importlib
import inspect
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional, Tuple, Union
import numpy as np
import pandas
import yaml
import mlflow
import mlflow.pyfunc.loaders
import mlflow.pyfunc.model
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import _DATABRICKS_FS_LOADER_MODULE, MLMODEL_FILE_NAME
from mlflow.models.signature import (
from mlflow.models.utils import (
from mlflow.protos.databricks_pb2 import (
from mlflow.pyfunc.model import (
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import (
from mlflow.utils import (
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils._spark_utils import modified_environ
from mlflow.utils.annotations import deprecated, developer_stable, experimental
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import (
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.requirements_utils import (
def _get_model_dependencies(model_uri, format='pip'):
    model_dir = _download_artifact_from_uri(model_uri)

    def get_conda_yaml_path():
        model_config = _get_flavor_configuration_from_ml_model_file(os.path.join(model_dir, MLMODEL_FILE_NAME), flavor_name=FLAVOR_NAME)
        return os.path.join(model_dir, _extract_conda_env(model_config[ENV]))
    if format == 'pip':
        requirements_file = os.path.join(model_dir, _REQUIREMENTS_FILE_NAME)
        if os.path.exists(requirements_file):
            return requirements_file
        _logger.info(f"{_REQUIREMENTS_FILE_NAME} is not found in the model directory. Falling back to extracting pip requirements from the model's 'conda.yaml' file. Conda dependencies will be ignored.")
        with open(get_conda_yaml_path()) as yf:
            conda_yaml = yaml.safe_load(yf)
        conda_deps = conda_yaml.get('dependencies', [])
        for index, dep in enumerate(conda_deps):
            if isinstance(dep, dict) and 'pip' in dep:
                pip_deps_index = index
                break
        else:
            raise MlflowException('No pip section found in conda.yaml file in the model directory.', error_code=RESOURCE_DOES_NOT_EXIST)
        pip_deps = conda_deps.pop(pip_deps_index)['pip']
        tmp_dir = tempfile.mkdtemp()
        pip_file_path = os.path.join(tmp_dir, _REQUIREMENTS_FILE_NAME)
        with open(pip_file_path, 'w') as f:
            f.write('\n'.join(pip_deps) + '\n')
        if len(conda_deps) > 0:
            _logger.warning(f'The following conda dependencies have been excluded from the environment file: {', '.join(conda_deps)}.')
        return pip_file_path
    elif format == 'conda':
        return get_conda_yaml_path()
    else:
        raise MlflowException(f"Illegal format argument '{format}'.", error_code=INVALID_PARAMETER_VALUE)