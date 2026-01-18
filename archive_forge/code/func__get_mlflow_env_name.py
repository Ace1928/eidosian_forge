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
def _get_mlflow_env_name(s):
    """Creates an environment name for an MLflow model by hashing the given string.

    Args:
        s: String to hash (e.g. the content of `conda.yaml`).

    Returns:
        String in the form of "mlflow-{hash}"
        (e.g. "mlflow-da39a3ee5e6b4b0d3255bfef95601890afd80709")

    """
    return 'mlflow-' + insecure_hash.sha1(s.encode('utf-8')).hexdigest()