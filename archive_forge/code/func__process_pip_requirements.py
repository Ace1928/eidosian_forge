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
def _process_pip_requirements(default_pip_requirements, pip_requirements=None, extra_pip_requirements=None):
    """
    Processes `pip_requirements` and `extra_pip_requirements` passed to `mlflow.*.save_model` or
    `mlflow.*.log_model`, and returns a tuple of (conda_env, pip_requirements, pip_constraints).
    """
    constraints = []
    if pip_requirements is not None:
        pip_reqs, constraints = _parse_pip_requirements(pip_requirements)
    elif extra_pip_requirements is not None:
        extra_pip_requirements, constraints = _parse_pip_requirements(extra_pip_requirements)
        pip_reqs = default_pip_requirements + extra_pip_requirements
    else:
        pip_reqs = default_pip_requirements
    if not _contains_mlflow_requirement(pip_reqs):
        pip_reqs.insert(0, _generate_mlflow_version_pinning())
    sanitized_pip_reqs = _deduplicate_requirements(pip_reqs)
    warn_dependency_requirement_mismatches(sanitized_pip_reqs)
    if constraints:
        sanitized_pip_reqs.append(f'-c {_CONSTRAINTS_FILE_NAME}')
    conda_env = _mlflow_conda_env(additional_pip_deps=sanitized_pip_reqs, install_mlflow=False)
    return (conda_env, sanitized_pip_reqs, constraints)