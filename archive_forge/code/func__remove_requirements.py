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
def _remove_requirements(reqs_to_remove: List[Requirement], old_reqs: List[Requirement]) -> List[str]:
    old_reqs_dict = {req.name: str(req) for req in old_reqs}
    for req in reqs_to_remove:
        if req.name not in old_reqs_dict:
            _logger.warning(f'"{req.name}" not found in requirements, ignoring')
        old_reqs_dict.pop(req.name, None)
    return list(old_reqs_dict.values())