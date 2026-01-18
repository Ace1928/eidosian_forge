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
def _process_conda_env(conda_env):
    """
    Processes `conda_env` passed to `mlflow.*.save_model` or `mlflow.*.log_model`, and returns
    a tuple of (conda_env, pip_requirements, pip_constraints).
    """
    if isinstance(conda_env, str):
        with open(conda_env) as f:
            conda_env = yaml.safe_load(f)
    elif not isinstance(conda_env, dict):
        raise TypeError(f'Expected a string path to a conda env yaml file or a `dict` representing a conda env, but got `{type(conda_env).__name__}`')
    pip_reqs = _get_pip_deps(conda_env)
    pip_reqs, constraints = _parse_pip_requirements(pip_reqs)
    if not _contains_mlflow_requirement(pip_reqs):
        pip_reqs.insert(0, _generate_mlflow_version_pinning())
    warn_dependency_requirement_mismatches(pip_reqs)
    if constraints:
        pip_reqs.append(f'-c {_CONSTRAINTS_FILE_NAME}')
    conda_env = _overwrite_pip_deps(conda_env, pip_reqs)
    return (conda_env, pip_reqs, constraints)