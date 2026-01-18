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
def _get_installed_version(package, module=None):
    """
    Obtains the installed package version using `importlib_metadata.version`. If it fails, use
    `__import__(module or package).__version__`.
    """
    try:
        version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        version = __import__(module or package).__version__
    if package == 'pyspark':
        version = _strip_dev_version_suffix(version)
    return version