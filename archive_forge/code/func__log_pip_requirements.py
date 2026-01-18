import logging
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from typing import List
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import Version
from mlflow.environment_variables import _MLFLOW_TESTING, MLFLOW_INPUT_EXAMPLE_INFERENCE_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import PYTHON_VERSION, insecure_hash
from mlflow.utils.os import is_windows
from mlflow.utils.process import _exec_cmd
from mlflow.utils.requirements_utils import (
from mlflow.utils.timeout import MlflowTimeoutError, run_with_timeout
from mlflow.version import VERSION
def _log_pip_requirements(conda_env, path, requirements_file=_REQUIREMENTS_FILE_NAME):
    pip_deps = _get_pip_deps(conda_env)
    _mlflow_additional_pip_env(pip_deps, path=os.path.join(path, requirements_file))