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
def __init_from_csc(self, csc: scipy.sparse.csc_matrix, params_str: str, ref_dataset: Optional[_DatasetHandle]) -> 'Dataset':
    """Initialize data from a CSC matrix."""
    if len(csc.indices) != len(csc.data):
        raise ValueError(f'Length mismatch: {len(csc.indices)} vs {len(csc.data)}')
    self._handle = ctypes.c_void_p()
    ptr_indptr, type_ptr_indptr, __ = _c_int_array(csc.indptr)
    ptr_data, type_ptr_data, _ = _c_float_array(csc.data)
    assert csc.shape[0] <= _MAX_INT32
    csc_indices = csc.indices.astype(np.int32, copy=False)
    _safe_call(_LIB.LGBM_DatasetCreateFromCSC(ptr_indptr, ctypes.c_int(type_ptr_indptr), csc_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int64(len(csc.indptr)), ctypes.c_int64(len(csc.data)), ctypes.c_int64(csc.shape[0]), _c_str(params_str), ref_dataset, ctypes.byref(self._handle)))
    return self