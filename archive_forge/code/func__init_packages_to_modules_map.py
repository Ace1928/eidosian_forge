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
def _init_packages_to_modules_map():
    _init_modules_to_packages_map()
    global _PACKAGES_TO_MODULES
    _PACKAGES_TO_MODULES = {}
    for module, pkg_list in _MODULES_TO_PACKAGES.items():
        for pkg_name in pkg_list:
            _PACKAGES_TO_MODULES[pkg_name] = module