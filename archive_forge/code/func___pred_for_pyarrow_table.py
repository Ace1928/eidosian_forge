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
def __pred_for_pyarrow_table(self, table: pa_Table, start_iteration: int, num_iteration: int, predict_type: int) -> Tuple[np.ndarray, int]:
    """Predict for a PyArrow table."""
    if not PYARROW_INSTALLED:
        raise LightGBMError('Cannot predict from Arrow without `pyarrow` installed.')
    if not all((arrow_is_integer(t) or arrow_is_floating(t) for t in table.schema.types)):
        raise ValueError('Arrow table may only have integer or floating point datatypes')
    n_preds = self.__get_num_preds(start_iteration=start_iteration, num_iteration=num_iteration, nrow=table.num_rows, predict_type=predict_type)
    preds = np.empty(n_preds, dtype=np.float64)
    out_num_preds = ctypes.c_int64(0)
    c_array = _export_arrow_to_c(table)
    _safe_call(_LIB.LGBM_BoosterPredictForArrow(self._handle, ctypes.c_int64(c_array.n_chunks), ctypes.c_void_p(c_array.chunks_ptr), ctypes.c_void_p(c_array.schema_ptr), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), _c_str(self.pred_parameter), ctypes.byref(out_num_preds), preds.ctypes.data_as(ctypes.POINTER(ctypes.c_double))))
    if n_preds != out_num_preds.value:
        raise ValueError('Wrong length for predict results')
    return (preds, table.num_rows)