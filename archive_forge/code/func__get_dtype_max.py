from __future__ import annotations
import functools
import itertools
from typing import (
import warnings
import numpy as np
from pandas._config import get_option
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
def _get_dtype_max(dtype: np.dtype) -> np.dtype:
    dtype_max = dtype
    if dtype.kind in 'bi':
        dtype_max = np.dtype(np.int64)
    elif dtype.kind == 'u':
        dtype_max = np.dtype(np.uint64)
    elif dtype.kind == 'f':
        dtype_max = np.dtype(np.float64)
    return dtype_max