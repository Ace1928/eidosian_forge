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
def _is_local_scheme(paths: Union[str, List[str]]) -> bool:
    """Returns True if the given paths are in local scheme.
    Note: The paths must be in same scheme, i.e. it's invalid and
    will raise error if paths are mixed with different schemes.
    """
    if isinstance(paths, str):
        paths = [paths]
    if isinstance(paths, pathlib.Path):
        paths = [str(paths)]
    elif not isinstance(paths, list) or any((not isinstance(p, str) for p in paths)):
        raise ValueError('paths must be a path string or a list of path strings.')
    elif len(paths) == 0:
        raise ValueError('Must provide at least one path.')
    num = sum((urllib.parse.urlparse(path).scheme == _LOCAL_SCHEME for path in paths))
    if num > 0 and num < len(paths):
        raise ValueError(f'The paths must all be local-scheme or not local-scheme, but found mixed {paths}')
    return num == len(paths)