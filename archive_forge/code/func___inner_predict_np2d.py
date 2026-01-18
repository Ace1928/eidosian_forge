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
def __inner_predict_np2d(self, mat: np.ndarray, start_iteration: int, num_iteration: int, predict_type: int, preds: Optional[np.ndarray]) -> Tuple[np.ndarray, int]:
    if mat.dtype == np.float32 or mat.dtype == np.float64:
        data = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
    else:
        data = np.array(mat.reshape(mat.size), dtype=np.float32)
    ptr_data, type_ptr_data, _ = _c_float_array(data)
    n_preds = self.__get_num_preds(start_iteration=start_iteration, num_iteration=num_iteration, nrow=mat.shape[0], predict_type=predict_type)
    if preds is None:
        preds = np.empty(n_preds, dtype=np.float64)
    elif len(preds.shape) != 1 or len(preds) != n_preds:
        raise ValueError('Wrong length of pre-allocated predict array')
    out_num_preds = ctypes.c_int64(0)
    _safe_call(_LIB.LGBM_BoosterPredictForMat(self._handle, ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int32(mat.shape[0]), ctypes.c_int32(mat.shape[1]), ctypes.c_int(_C_API_IS_ROW_MAJOR), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.byref(out_num_preds), preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
    if n_preds != out_num_preds.value:
        raise ValueError('Wrong length for predict results')
    return (preds, mat.shape[0])