import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def _meta_from_pandas_series(data: DataType, name: str, dtype: Optional[NumpyDType], handle: ctypes.c_void_p) -> None:
    """Help transform pandas series for meta data like labels"""
    data = data.values.astype('float')
    if is_pd_sparse_dtype(getattr(data, 'dtype', data)):
        data = data.to_dense()
    assert len(data.shape) == 1 or data.shape[1] == 0 or data.shape[1] == 1
    _meta_from_numpy(data, name, dtype, handle)