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
def _strip_local_version_label(version):
    """Strips a local version label in `version`.

    Local version identifiers:
    https://www.python.org/dev/peps/pep-0440/#local-version-identifiers

    Args:
        version: A version string to strip.
    """

    class IgnoreLocal(Version):

        @property
        def local(self):
            return None
    try:
        return str(IgnoreLocal(version))
    except InvalidVersion:
        return version