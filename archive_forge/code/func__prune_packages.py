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
def _prune_packages(packages):
    """
    Prunes packages required by other packages. For example, `["scikit-learn", "numpy"]` is pruned
    to `["scikit-learn"]`.
    """
    packages = set(packages)
    requires = set(_flatten(map(_get_requires_recursive, packages)))
    return packages - (requires - set(_get_requires('mlflow')))