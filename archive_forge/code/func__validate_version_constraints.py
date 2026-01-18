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
def _validate_version_constraints(requirements):
    """
    Validates the version constraints of given Python package requirements using pip's resolver with
    the `--dry-run` option enabled that performs validation only (will not install packages).

    This function writes the requirements to a temporary file and then attempts to resolve
    them using pip's `--dry-run` install option. If any version conflicts are detected, it
    raises an MlflowException with details of the conflict.

    Args:
        requirements (list of str): A list of package requirements (e.g., `["pandas>=1.15",
        "pandas<2"]`).

    Raises:
        MlflowException: If any version conflicts are detected among the provided requirements.

    Returns:
        None: This function does not return anything. It either completes successfully or raises
        an MlflowException.

    Example:
        _validate_version_constraints(["tensorflow<2.0", "tensorflow>2.3"])
        # This will raise an exception due to boundary validity.
    """
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        tmp_file.write('\n'.join(requirements))
        tmp_file_name = tmp_file.name
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--dry-run', '-r', tmp_file_name], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise MlflowException.invalid_parameter_value(f'The specified requirements versions are incompatible. Detected conflicts: \n{e.stderr.decode()}')
    finally:
        os.remove(tmp_file_name)