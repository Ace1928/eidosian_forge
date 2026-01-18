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
def _mlflow_conda_env(path=None, additional_conda_deps=None, additional_pip_deps=None, additional_conda_channels=None, install_mlflow=True):
    """Creates a Conda environment with the specified package channels and dependencies. If there
    are any pip dependencies, including from the install_mlflow parameter, then pip will be added to
    the conda dependencies. This is done to ensure that the pip inside the conda environment is
    used to install the pip dependencies.

    Args:
        path: Local filesystem path where the conda env file is to be written. If unspecified,
            the conda env will not be written to the filesystem; it will still be returned
            in dictionary format.
        additional_conda_deps: List of additional conda dependencies passed as strings.
        additional_pip_deps: List of additional pip dependencies passed as strings.
        additional_conda_channels: List of additional conda channels to search when resolving
            packages.

    Returns:
        None if path is specified. Otherwise, the a dictionary representation of the
        Conda environment.

    """
    additional_pip_deps = additional_pip_deps or []
    mlflow_deps = [f'mlflow=={VERSION}'] if install_mlflow and (not _contains_mlflow_requirement(additional_pip_deps)) else []
    pip_deps = mlflow_deps + additional_pip_deps
    conda_deps = additional_conda_deps if additional_conda_deps else []
    if pip_deps:
        pip_version = _get_pip_version()
        if pip_version is not None:
            conda_deps.append(f'pip<={pip_version}')
        else:
            _logger.warning('Failed to resolve installed pip version. ``pip`` will be added to conda.yaml environment spec without a version specifier.')
            conda_deps.append('pip')
    env = yaml.safe_load(_conda_header)
    env['dependencies'] = [f'python={PYTHON_VERSION}']
    env['dependencies'] += conda_deps
    env['dependencies'].append({'pip': pip_deps})
    if additional_conda_channels is not None:
        env['channels'] += additional_conda_channels
    if path is not None:
        with open(path, 'w') as out:
            yaml.safe_dump(env, stream=out, default_flow_style=False)
        return None
    else:
        return env