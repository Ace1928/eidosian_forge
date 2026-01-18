import ctypes
import logging
import os
import pathlib
import posixpath
import shlex
import signal
import subprocess
import sys
import warnings
from pathlib import Path
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import FlavorBackend, docker_utils
from mlflow.models.docker_utils import PYTHON_SLIM_BASE_IMAGE, UBUNTU_BASE_IMAGE
from mlflow.models.model import MLMODEL_FILE_NAME, Model
from mlflow.pyfunc import (
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils.conda import get_conda_bin_executable, get_or_create_conda_env
from mlflow.utils.environment import Environment, _PythonEnv
from mlflow.utils.file_utils import (
from mlflow.utils.model_utils import _get_all_flavor_configurations
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.os import is_windows
from mlflow.utils.process import ShellCommandException, cache_return_value_per_process
from mlflow.utils.virtualenv import (
from mlflow.version import VERSION
@cache_return_value_per_process
def _get_or_create_env_root_dir(should_use_nfs):
    if should_use_nfs:
        root_tmp_dir = get_or_create_nfs_tmp_dir()
    else:
        root_tmp_dir = get_or_create_tmp_dir()
    env_root_dir = os.path.join(root_tmp_dir, 'envs')
    os.makedirs(env_root_dir, exist_ok=True)
    return env_root_dir