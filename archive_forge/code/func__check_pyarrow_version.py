import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def _check_pyarrow_version():
    """Check that pyarrow's version is within the supported bounds."""
    global _VERSION_VALIDATED
    if not _VERSION_VALIDATED:
        if os.environ.get(RAY_DISABLE_PYARROW_VERSION_CHECK, '0') == '1':
            _VERSION_VALIDATED = True
            return
        version = _get_pyarrow_version()
        if version is not None:
            from pkg_resources._vendor.packaging.version import parse as parse_version
            if parse_version(version) < parse_version(MIN_PYARROW_VERSION):
                raise ImportError(f'Dataset requires pyarrow >= {MIN_PYARROW_VERSION}, but {version} is installed. Reinstall with `pip install -U "pyarrow"`. If you want to disable this pyarrow version check, set the environment variable {RAY_DISABLE_PYARROW_VERSION_CHECK}=1.')
        else:
            logger.warning(f"You are using the 'pyarrow' module, but the exact version is unknown (possibly carried as an internal component by another module). Please make sure you are using pyarrow >= {MIN_PYARROW_VERSION} to ensure compatibility with Ray Dataset. If you want to disable this pyarrow version check, set the environment variable {RAY_DISABLE_PYARROW_VERSION_CHECK}=1.")
        _VERSION_VALIDATED = True