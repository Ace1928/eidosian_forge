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
def _push_rows(self, data: np.ndarray) -> 'Dataset':
    """Add rows to Dataset.

        Parameters
        ----------
        data : numpy 1-D array
            New data to add to the Dataset.

        Returns
        -------
        self : Dataset
            Dataset object.
        """
    nrow, ncol = data.shape
    data = data.reshape(data.size)
    data_ptr, data_type, _ = _c_float_array(data)
    _safe_call(_LIB.LGBM_DatasetPushRows(self._handle, data_ptr, data_type, ctypes.c_int32(nrow), ctypes.c_int32(ncol), ctypes.c_int32(self._start_row)))
    self._start_row += nrow
    return self