import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def _convert_from_sliced_object(data: np.ndarray) -> np.ndarray:
    """Fix the memory of multi-dimensional sliced object."""
    if isinstance(data, np.ndarray) and isinstance(data.base, np.ndarray):
        if not data.flags.c_contiguous:
            _log_warning('Usage of np.ndarray subset (sliced data) is not recommended due to it will double the peak memory cost in LightGBM.')
            return np.copy(data)
    return data