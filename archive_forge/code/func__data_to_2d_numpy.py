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
def _data_to_2d_numpy(data: Any, dtype: 'np.typing.DTypeLike', name: str) -> np.ndarray:
    """Convert data to numpy 2-D array."""
    if _is_numpy_2d_array(data):
        return _cast_numpy_array_to_dtype(data, dtype)
    if _is_2d_list(data):
        return np.array(data, dtype=dtype)
    if isinstance(data, pd_DataFrame):
        _check_for_bad_pandas_dtypes(data.dtypes)
        return _cast_numpy_array_to_dtype(data.values, dtype)
    raise TypeError(f'Wrong type({type(data).__name__}) for {name}.\nIt should be list of lists, numpy 2-D array or pandas DataFrame')