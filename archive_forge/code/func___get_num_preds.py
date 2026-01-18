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
def __get_num_preds(self, start_iteration: int, num_iteration: int, nrow: int, predict_type: int) -> int:
    """Get size of prediction result."""
    if nrow > _MAX_INT32:
        raise LightGBMError(f'LightGBM cannot perform prediction for data with number of rows greater than MAX_INT32 ({_MAX_INT32}).\nYou can split your data into chunks and then concatenate predictions for them')
    n_preds = ctypes.c_int64(0)
    _safe_call(_LIB.LGBM_BoosterCalcNumPredict(self._handle, ctypes.c_int(nrow), ctypes.c_int(predict_type), ctypes.c_int(start_iteration), ctypes.c_int(num_iteration), ctypes.byref(n_preds)))
    return n_preds.value