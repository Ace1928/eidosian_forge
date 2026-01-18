import textwrap
import warnings
from functools import wraps
from typing import Dict
import importlib_metadata
from packaging.version import Version
from mlflow.ml_package_versions import _ML_PACKAGE_VERSIONS
from mlflow.utils.autologging_utils.versioning import (
imports declared from a common root path if multiple files are defined with import dependencies
def _get_indentation_of_key(line: str, placeholder: str) -> str:
    index = line.find(placeholder)
    return index * ' ' if index != -1 else ''