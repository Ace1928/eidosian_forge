from __future__ import annotations
from collections import abc
from typing import (
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core import (
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.indexes.api import (
from pandas.core.internals.array_manager import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def _prep_ndarraylike(values, copy: bool=True) -> np.ndarray:
    if len(values) == 0:
        return np.empty((0, 0), dtype=object)
    elif isinstance(values, range):
        arr = range_to_ndarray(values)
        return arr[..., np.newaxis]

    def convert(v):
        if not is_list_like(v) or isinstance(v, ABCDataFrame):
            return v
        v = extract_array(v, extract_numpy=True)
        res = maybe_convert_platform(v)
        return res
    if is_list_like(values[0]):
        values = np.array([convert(v) for v in values])
    elif isinstance(values[0], np.ndarray) and values[0].ndim == 0:
        values = np.array([convert(v) for v in values])
    else:
        values = convert(values)
    return _ensure_2d(values)