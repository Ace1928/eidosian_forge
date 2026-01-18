from __future__ import annotations
import functools
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import na_value_for_dtype
from pandas.core.construction import ensure_wrapped_if_datetimelike
def _convert_wrapper(f, conv_dtype):

    def wrapper(arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan) -> None:
        if conv_dtype == object:
            arr = ensure_wrapped_if_datetimelike(arr)
        arr = arr.astype(conv_dtype)
        f(arr, indexer, out, fill_value=fill_value)
    return wrapper