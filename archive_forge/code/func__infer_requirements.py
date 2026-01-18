import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections import namedtuple
from itertools import chain, filterfalse
from pathlib import Path
from threading import Timer
from typing import List, NamedTuple, Optional
import importlib_metadata
import pkg_resources  # noqa: TID251
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version
import mlflow
from mlflow.environment_variables import MLFLOW_REQUIREMENTS_INFERENCE_TIMEOUT
from mlflow.exceptions import MlflowException
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils.versioning import _strip_dev_version_suffix
from mlflow.utils.databricks_utils import (
def _infer_requirements(model_uri, flavor):
    """Infers the pip requirements of the specified model by creating a subprocess and loading
    the model in it to determine which packages are imported.

    Args:
        model_uri: The URI of the model.
        flavor: The flavor name of the model.

    Returns:
        A list of inferred pip requirements.

    """
    _init_modules_to_packages_map()
    global _PYPI_PACKAGE_INDEX
    if _PYPI_PACKAGE_INDEX is None:
        _PYPI_PACKAGE_INDEX = _load_pypi_package_index()
    modules = _capture_imported_modules(model_uri, flavor)
    packages = _flatten([_MODULES_TO_PACKAGES.get(module, []) for module in modules])
    packages = map(_normalize_package_name, packages)
    packages = _prune_packages(packages)
    excluded_packages = ['setuptools', *_MODULES_TO_PACKAGES.get('mlflow', [])]
    packages = packages - set(excluded_packages)
    unrecognized_packages = packages - _PYPI_PACKAGE_INDEX.package_names - {'mlflow[gateway]'}
    if unrecognized_packages:
        _logger.warning('The following packages were not found in the public PyPI package index as of %s; if these packages are not present in the public PyPI index, you must install them manually before loading your model: %s', _PYPI_PACKAGE_INDEX.date, unrecognized_packages)
    return sorted(map(_get_pinned_requirement, packages))