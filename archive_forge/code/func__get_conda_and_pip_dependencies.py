import os
import tempfile
import types
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Optional
import numpy as np
import yaml
import mlflow
import mlflow.utils.autologging_utils
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
from mlflow.utils.requirements_utils import _get_package_name
from mlflow.utils.uri import append_to_uri_path
def _get_conda_and_pip_dependencies(conda_env):
    """
    Extract conda and pip dependencies from conda environments

    Args:
        conda_env: Conda environment
    """
    conda_deps = []
    pip_deps_set = set()
    for dependency in conda_env['dependencies']:
        if isinstance(dependency, dict) and dependency['pip']:
            for pip_dependency in dependency['pip']:
                if pip_dependency != 'mlflow':
                    pip_deps_set.add(pip_dependency)
        else:
            package_name = _get_package_name(dependency)
            if package_name is not None and package_name not in ['python', 'pip']:
                conda_deps.append(dependency)
    return (conda_deps, sorted(pip_deps_set))