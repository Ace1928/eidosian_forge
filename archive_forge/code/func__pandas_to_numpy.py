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
def _pandas_to_numpy(data: pd_DataFrame, target_dtype: 'np.typing.DTypeLike') -> np.ndarray:
    _check_for_bad_pandas_dtypes(data.dtypes)
    try:
        return data.to_numpy(dtype=target_dtype, copy=False)
    except TypeError:
        return data.astype(target_dtype, copy=False).values
    except ValueError:
        return data.to_numpy(dtype=target_dtype, na_value=np.nan)