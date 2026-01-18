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
def _save_model_with_loader_module_and_data_path(path, loader_module, data_path=None, code_paths=None, conda_env=None, mlflow_model=None, pip_requirements=None, extra_pip_requirements=None, model_config=None):
    """
    Export model as a generic Python function model.

    Args:
        path: The path to which to save the Python model.
        loader_module: The name of the Python module that is used to load the model
            from ``data_path``. This module must define a method with the prototype
            ``_load_pyfunc(data_path)``.
        data_path: Path to a file or directory containing model data.
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path before the model is loaded.
        conda_env: Either a dictionary representation of a Conda environment or the path to a
            Conda environment yaml file. If provided, this describes the environment
            this model should be run in.

    Returns:
        Model configuration containing model info.
    """
    data = None
    if data_path is not None:
        model_file = _copy_file_or_tree(src=data_path, dst=path, dst_dir='data')
        data = model_file
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)
    if mlflow_model is None:
        mlflow_model = Model()
    mlflow.pyfunc.add_to_model(mlflow_model, loader_module=loader_module, code=code_dir_subpath, data=data, conda_env=_CONDA_ENV_FILE_NAME, python_env=_PYTHON_ENV_FILE_NAME, model_config=model_config)
    if (size := get_total_file_size(path)):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))
    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            inferred_reqs = mlflow.models.infer_pip_requirements(path, FLAVOR_NAME, fallback=default_reqs)
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(default_reqs, pip_requirements, extra_pip_requirements)
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)
    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), 'w') as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), '\n'.join(pip_constraints))
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), '\n'.join(pip_requirements))
    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))
    return mlflow_model