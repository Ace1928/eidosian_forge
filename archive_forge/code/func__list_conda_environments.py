import json
import logging
import os
import yaml
from mlflow.environment_variables import MLFLOW_CONDA_CREATE_ENV_CMD, MLFLOW_CONDA_HOME
from mlflow.exceptions import ExecutionException
from mlflow.utils import insecure_hash, process
from mlflow.utils.environment import Environment
from mlflow.utils.os import is_windows
def _list_conda_environments(extra_env=None):
    """Return a list of names of conda environments.

    Args:
        extra_env: Extra environment variables for running "conda env list" command.

    """
    prc = process._exec_cmd([get_conda_bin_executable('conda'), 'env', 'list', '--json'], extra_env=extra_env)
    return list(map(os.path.basename, json.loads(prc.stdout).get('envs', [])))