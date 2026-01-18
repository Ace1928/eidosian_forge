import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _ensure_np_dtype(data: DataType, dtype: Optional[NumpyDType]) -> Tuple[np.ndarray, Optional[NumpyDType]]:
    if data.dtype.hasobject or data.dtype in [np.float16, np.bool_]:
        dtype = np.float32
        data = data.astype(dtype, copy=False)
    if not data.flags.aligned:
        data = np.require(data, requirements='A')
    return (data, dtype)