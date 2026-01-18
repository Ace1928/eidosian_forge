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
def __init_from_np2d(self, mat: np.ndarray, params_str: str, ref_dataset: Optional[_DatasetHandle]) -> 'Dataset':
    """Initialize data from a 2-D numpy matrix."""
    if len(mat.shape) != 2:
        raise ValueError('Input numpy.ndarray must be 2 dimensional')
    self._handle = ctypes.c_void_p()
    if mat.dtype == np.float32 or mat.dtype == np.float64:
        data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
    else:
        data = np.array(mat.reshape(mat.size), dtype=np.float32)
    ptr_data, type_ptr_data, _ = _c_float_array(data)
    _safe_call(_LIB.LGBM_DatasetCreateFromMat(ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int32(mat.shape[0]), ctypes.c_int32(mat.shape[1]), ctypes.c_int(_C_API_IS_ROW_MAJOR), _c_str(params_str), ref_dataset, ctypes.byref(self._handle)))
    return self