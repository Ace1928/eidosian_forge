import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _maybe_np_slice(data: DataType, dtype: Optional[NumpyDType]) -> np.ndarray:
    """Handle numpy slice.  This can be removed if we use __array_interface__."""
    try:
        if not data.flags.c_contiguous:
            data = np.array(data, copy=True, dtype=dtype)
        else:
            data = np.array(data, copy=False, dtype=dtype)
    except AttributeError:
        data = np.array(data, copy=False, dtype=dtype)
    data, dtype = _ensure_np_dtype(data, dtype)
    return data