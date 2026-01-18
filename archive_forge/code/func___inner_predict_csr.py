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
def __inner_predict_csr(self, csr: scipy.sparse.csr_matrix, start_iteration: int, num_iteration: int, predict_type: int, preds: Optional[np.ndarray]) -> Tuple[np.ndarray, int]:
    nrow = len(csr.indptr) - 1
    n_preds = self.__get_num_preds(start_iteration=start_iteration, num_iteration=num_iteration, nrow=nrow, predict_type=predict_type)
    if preds is None:
        preds = np.empty(n_preds, dtype=np.float64)
    elif len(preds.shape) != 1 or len(preds) != n_preds:
        raise ValueError('Wrong length of pre-allocated predict array')
    out_num_preds = ctypes.c_int64(0)
    ptr_indptr, type_ptr_indptr, _ = _c_int_array(csr.indptr)
    ptr_data, type_ptr_data, _ = _c_float_array(csr.data)
    assert csr.shape[1] <= _MAX_INT32
    csr_indices = csr.indices.astype(np.int32, copy=False)
    _safe_call(_LIB.LGBM_BoosterPredictForCSR(self._handle, ptr_indptr, ctypes.c_int(type_ptr_indptr), csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int64(len(csr.indptr)), ctypes.c_int64(len(csr.data)), ctypes.c_int64(csr.shape[1]), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.byref(out_num_preds), preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
    if n_preds != out_num_preds.value:
        raise ValueError('Wrong length for predict results')
    return (preds, nrow)