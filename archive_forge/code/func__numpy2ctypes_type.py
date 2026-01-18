import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def _numpy2ctypes_type(dtype: Type[np.number]) -> Type[CNumeric]:
    _NUMPY_TO_CTYPES_MAPPING: Dict[Type[np.number], Type[CNumeric]] = {np.float32: ctypes.c_float, np.float64: ctypes.c_double, np.uint32: ctypes.c_uint, np.uint64: ctypes.c_uint64, np.int32: ctypes.c_int32, np.int64: ctypes.c_int64}
    if np.intc is not np.int32:
        _NUMPY_TO_CTYPES_MAPPING[np.intc] = _NUMPY_TO_CTYPES_MAPPING[np.int32]
    if dtype not in _NUMPY_TO_CTYPES_MAPPING:
        raise TypeError(f'Supported types: {_NUMPY_TO_CTYPES_MAPPING.keys()}, got: {dtype}')
    return _NUMPY_TO_CTYPES_MAPPING[dtype]