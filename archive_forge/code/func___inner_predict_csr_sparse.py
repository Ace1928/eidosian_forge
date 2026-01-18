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
def __inner_predict_csr_sparse(self, csr: scipy.sparse.csr_matrix, start_iteration: int, num_iteration: int, predict_type: int) -> Tuple[Union[List[scipy.sparse.csc_matrix], List[scipy.sparse.csr_matrix]], int]:
    ptr_indptr, type_ptr_indptr, __ = _c_int_array(csr.indptr)
    ptr_data, type_ptr_data, _ = _c_float_array(csr.data)
    csr_indices = csr.indices.astype(np.int32, copy=False)
    matrix_type = _C_API_MATRIX_TYPE_CSR
    out_ptr_indptr: _ctypes_int_ptr
    if type_ptr_indptr == _C_API_DTYPE_INT32:
        out_ptr_indptr = ctypes.POINTER(ctypes.c_int32)()
    else:
        out_ptr_indptr = ctypes.POINTER(ctypes.c_int64)()
    out_ptr_indices = ctypes.POINTER(ctypes.c_int32)()
    out_ptr_data: _ctypes_float_ptr
    if type_ptr_data == _C_API_DTYPE_FLOAT32:
        out_ptr_data = ctypes.POINTER(ctypes.c_float)()
    else:
        out_ptr_data = ctypes.POINTER(ctypes.c_double)()
    out_shape = np.empty(2, dtype=np.int64)
    _safe_call(_LIB.LGBM_BoosterPredictSparseOutput(self._handle, ptr_indptr, ctypes.c_int(type_ptr_indptr), csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int64(len(csr.indptr)), ctypes.c_int64(len(csr.data)), ctypes.c_int64(csr.shape[1]), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.c_int(matrix_type), out_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.byref(out_ptr_indptr), ctypes.byref(out_ptr_indices), ctypes.byref(out_ptr_data)))
    matrices = self.__create_sparse_native(cs=csr, out_shape=out_shape, out_ptr_indptr=out_ptr_indptr, out_ptr_indices=out_ptr_indices, out_ptr_data=out_ptr_data, indptr_type=type_ptr_indptr, data_type=type_ptr_data, is_csr=True)
    nrow = len(csr.indptr) - 1
    return (matrices, nrow)