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
class _MismatchedPackageInfo(NamedTuple):
    package_name: str
    installed_version: Optional[str]
    requirement: str

    def __str__(self):
        current_status = self.installed_version if self.installed_version else 'uninstalled'
        return f'{self.package_name} (current: {current_status}, required: {self.requirement})'