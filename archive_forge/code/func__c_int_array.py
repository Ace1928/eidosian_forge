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
def _c_int_array(data: np.ndarray) -> Tuple[_ctypes_int_ptr, int, np.ndarray]:
    """Get pointer of int numpy array / list."""
    if _is_1d_list(data):
        data = np.array(data, copy=False)
    if _is_numpy_1d_array(data):
        data = _convert_from_sliced_object(data)
        assert data.flags.c_contiguous
        ptr_data: _ctypes_int_ptr
        if data.dtype == np.int32:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            type_data = _C_API_DTYPE_INT32
        elif data.dtype == np.int64:
            ptr_data = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
            type_data = _C_API_DTYPE_INT64
        else:
            raise TypeError(f'Expected np.int32 or np.int64, met type({data.dtype})')
    else:
        raise TypeError(f'Unknown type({type(data).__name__})')
    return (ptr_data, type_data, data)