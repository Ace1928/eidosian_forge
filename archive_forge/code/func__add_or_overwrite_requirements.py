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
def _add_or_overwrite_requirements(new_reqs: List[Requirement], old_reqs: List[Requirement]) -> List[str]:
    deduped_new_reqs = _deduplicate_requirements([str(req) for req in new_reqs])
    deduped_new_reqs = [Requirement(req) for req in deduped_new_reqs]
    old_reqs_dict = {req.name: str(req) for req in old_reqs}
    new_reqs_dict = {req.name: str(req) for req in deduped_new_reqs}
    old_reqs_dict.update(new_reqs_dict)
    return list(old_reqs_dict.values())