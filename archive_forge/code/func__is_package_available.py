import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import warnings
from collections import OrderedDict
from functools import lru_cache
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union
from packaging import version
from . import logging
def _is_package_available(pkg_name: str, return_version: bool=False) -> Union[Tuple[bool, str], bool]:
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = 'N/A'
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logger.debug(f'Detected {pkg_name} version {package_version}')
    if return_version:
        return (package_exists, package_version)
    else:
        return package_exists