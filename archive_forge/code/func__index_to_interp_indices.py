from __future__ import annotations
from functools import wraps
from typing import (
import numpy as np
from pandas._libs import (
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.core.dtypes.cast import infer_dtype_from
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import DatetimeTZDtype
from pandas.core.dtypes.missing import (
def _index_to_interp_indices(index: Index, method: str) -> np.ndarray:
    """
    Convert Index to ndarray of indices to pass to NumPy/SciPy.
    """
    xarr = index._values
    if needs_i8_conversion(xarr.dtype):
        xarr = xarr.view('i8')
    if method == 'linear':
        inds = xarr
        inds = cast(np.ndarray, inds)
    else:
        inds = np.asarray(xarr)
        if method in ('values', 'index'):
            if inds.dtype == np.object_:
                inds = lib.maybe_convert_objects(inds)
    return inds