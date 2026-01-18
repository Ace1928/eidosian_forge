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
def _generate_mlflow_version_pinning() -> str:
    """Returns a pinned requirement for the current MLflow version (e.g., "mlflow==3.2.1").

    Returns:
        A pinned requirement for the current MLflow version.

    """
    if _MLFLOW_TESTING.get():
        return f'mlflow=={VERSION}'
    version = Version(VERSION)
    if not version.is_devrelease:
        return f'mlflow=={VERSION}'
    return f'mlflow=={version.major}.{version.minor}.{version.micro - 1}'