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
def __create_sparse_native(self, cs: Union[scipy.sparse.csc_matrix, scipy.sparse.csr_matrix], out_shape: np.ndarray, out_ptr_indptr: 'ctypes._Pointer', out_ptr_indices: 'ctypes._Pointer', out_ptr_data: 'ctypes._Pointer', indptr_type: int, data_type: int, is_csr: bool) -> Union[List[scipy.sparse.csc_matrix], List[scipy.sparse.csr_matrix]]:
    data_indices_len = out_shape[0]
    indptr_len = out_shape[1]
    if indptr_type == _C_API_DTYPE_INT32:
        out_indptr = _cint32_array_to_numpy(cptr=out_ptr_indptr, length=indptr_len)
    elif indptr_type == _C_API_DTYPE_INT64:
        out_indptr = _cint64_array_to_numpy(cptr=out_ptr_indptr, length=indptr_len)
    else:
        raise TypeError('Expected int32 or int64 type for indptr')
    if data_type == _C_API_DTYPE_FLOAT32:
        out_data = _cfloat32_array_to_numpy(cptr=out_ptr_data, length=data_indices_len)
    elif data_type == _C_API_DTYPE_FLOAT64:
        out_data = _cfloat64_array_to_numpy(cptr=out_ptr_data, length=data_indices_len)
    else:
        raise TypeError('Expected float32 or float64 type for data')
    out_indices = _cint32_array_to_numpy(cptr=out_ptr_indices, length=data_indices_len)
    per_class_indptr_shape = cs.indptr.shape[0]
    if not is_csr:
        per_class_indptr_shape += 1
    out_indptr_arrays = np.split(out_indptr, out_indptr.shape[0] / per_class_indptr_shape)
    cs_output_matrices = []
    offset = 0
    for cs_indptr in out_indptr_arrays:
        matrix_indptr_len = cs_indptr[cs_indptr.shape[0] - 1]
        cs_indices = out_indices[offset + cs_indptr[0]:offset + matrix_indptr_len]
        cs_data = out_data[offset + cs_indptr[0]:offset + matrix_indptr_len]
        offset += matrix_indptr_len
        cs_shape = [cs.shape[0], cs.shape[1] + 1]
        if is_csr:
            cs_output_matrices.append(scipy.sparse.csr_matrix((cs_data, cs_indices, cs_indptr), cs_shape))
        else:
            cs_output_matrices.append(scipy.sparse.csc_matrix((cs_data, cs_indices, cs_indptr), cs_shape))
    _safe_call(_LIB.LGBM_BoosterFreePredictSparse(out_ptr_indptr, out_ptr_indices, out_ptr_data, ctypes.c_int(indptr_type), ctypes.c_int(data_type)))
    if len(cs_output_matrices) == 1:
        return cs_output_matrices[0]
    return cs_output_matrices