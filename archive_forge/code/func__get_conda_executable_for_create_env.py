import json
import logging
import os
import yaml
from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException
from mlflow.utils import insecure_hash, process
from mlflow.utils.environment import Environment
from mlflow.utils.os import is_windows
def _get_conda_executable_for_create_env():
    """
    Returns the executable that should be used to create environments. This is "conda"
    by default, but it can be set to something else by setting the environment variable

    """
    return get_conda_bin_executable(MLFLOW_CONDA_CREATE_ENV_CMD.get())