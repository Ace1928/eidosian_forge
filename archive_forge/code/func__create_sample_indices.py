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
def _create_sample_indices(self, total_nrow: int) -> np.ndarray:
    """Get an array of randomly chosen indices from this ``Dataset``.

        Indices are sampled without replacement.

        Parameters
        ----------
        total_nrow : int
            Total number of rows to sample from.
            If this value is greater than the value of parameter ``bin_construct_sample_cnt``, only ``bin_construct_sample_cnt`` indices will be used.
            If Dataset has multiple input data, this should be the sum of rows of every file.

        Returns
        -------
        indices : numpy array
            Indices for sampled data.
        """
    param_str = _param_dict_to_str(self.get_params())
    sample_cnt = _get_sample_count(total_nrow, param_str)
    indices = np.empty(sample_cnt, dtype=np.int32)
    ptr_data, _, _ = _c_int_array(indices)
    actual_sample_cnt = ctypes.c_int32(0)
    _safe_call(_LIB.LGBM_SampleIndices(ctypes.c_int32(total_nrow), _c_str(param_str), ptr_data, ctypes.byref(actual_sample_cnt)))
    assert sample_cnt == actual_sample_cnt.value
    return indices