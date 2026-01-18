import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
def _get_version(package_name: str) -> str:
    return _package_versions.get(package_name, 'N/A')