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
def __pred_for_csr(self, csr: scipy.sparse.csr_matrix, start_iteration: int, num_iteration: int, predict_type: int) -> Tuple[np.ndarray, int]:
    """Predict for a CSR data."""
    if predict_type == _C_API_PREDICT_CONTRIB:
        return self.__inner_predict_csr_sparse(csr=csr, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type)
    nrow = len(csr.indptr) - 1
    if nrow > _MAX_INT32:
        sections = [0] + list(np.arange(start=_MAX_INT32, stop=nrow, step=_MAX_INT32)) + [nrow]
        n_preds = [self.__get_num_preds(start_iteration, num_iteration, i, predict_type) for i in np.diff(sections)]
        n_preds_sections = np.array([0] + n_preds, dtype=np.intp).cumsum()
        preds = np.empty(sum(n_preds), dtype=np.float64)
        for (start_idx, end_idx), (start_idx_pred, end_idx_pred) in zip(zip(sections, sections[1:]), zip(n_preds_sections, n_preds_sections[1:])):
            self.__inner_predict_csr(csr=csr[start_idx:end_idx], start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type, preds=preds[start_idx_pred:end_idx_pred])
        return (preds, nrow)
    else:
        return self.__inner_predict_csr(csr=csr, start_iteration=start_iteration, num_iteration=num_iteration, predict_type=predict_type, preds=None)