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
def _parse_pip_requirements(pip_requirements):
    """Parses an iterable of pip requirement strings or a pip requirements file.

    Args:
        pip_requirements: Either an iterable of pip requirement strings
            (e.g. ``["scikit-learn", "-r requirements.txt"]``) or the string path to a pip
            requirements file on the local filesystem (e.g. ``"requirements.txt"``). If ``None``,
            an empty list will be returned.

    Returns:
        A tuple of parsed requirements and constraints.
    """
    if pip_requirements is None:
        return ([], [])

    def _is_string(x):
        return isinstance(x, str)

    def _is_iterable(x):
        try:
            iter(x)
            return True
        except Exception:
            return False
    if _is_string(pip_requirements):
        with open(pip_requirements) as f:
            return _parse_pip_requirements(f.read().splitlines())
    elif _is_iterable(pip_requirements) and all(map(_is_string, pip_requirements)):
        requirements = []
        constraints = []
        for req_or_con in _parse_requirements(pip_requirements, is_constraint=False):
            if req_or_con.is_constraint:
                constraints.append(req_or_con.req_str)
            else:
                requirements.append(req_or_con.req_str)
        return (requirements, constraints)
    else:
        raise TypeError('`pip_requirements` must be either a string path to a pip requirements file on the local filesystem or an iterable of pip requirement strings, but got `{}`'.format(type(pip_requirements)))