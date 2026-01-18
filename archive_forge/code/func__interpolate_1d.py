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
def _interpolate_1d(indices: np.ndarray, yvalues: np.ndarray, method: str='linear', limit: int | None=None, limit_direction: str='forward', limit_area: Literal['inside', 'outside'] | None=None, fill_value: Any | None=None, bounds_error: bool=False, order: int | None=None, mask=None, **kwargs) -> None:
    """
    Logic for the 1-d interpolation.  The input
    indices and yvalues will each be 1-d arrays of the same length.

    Bounds_error is currently hardcoded to False since non-scipy ones don't
    take it as an argument.

    Notes
    -----
    Fills 'yvalues' in-place.
    """
    if mask is not None:
        invalid = mask
    else:
        invalid = isna(yvalues)
    valid = ~invalid
    if not valid.any():
        return
    if valid.all():
        return
    all_nans = set(np.flatnonzero(invalid))
    first_valid_index = find_valid_index(how='first', is_valid=valid)
    if first_valid_index is None:
        first_valid_index = 0
    start_nans = set(range(first_valid_index))
    last_valid_index = find_valid_index(how='last', is_valid=valid)
    if last_valid_index is None:
        last_valid_index = len(yvalues)
    end_nans = set(range(1 + last_valid_index, len(valid)))
    preserve_nans: list | set
    if limit_direction == 'forward':
        preserve_nans = start_nans | set(_interp_limit(invalid, limit, 0))
    elif limit_direction == 'backward':
        preserve_nans = end_nans | set(_interp_limit(invalid, 0, limit))
    else:
        preserve_nans = set(_interp_limit(invalid, limit, limit))
    if limit_area == 'inside':
        preserve_nans |= start_nans | end_nans
    elif limit_area == 'outside':
        mid_nans = all_nans - start_nans - end_nans
        preserve_nans |= mid_nans
    preserve_nans = sorted(preserve_nans)
    is_datetimelike = yvalues.dtype.kind in 'mM'
    if is_datetimelike:
        yvalues = yvalues.view('i8')
    if method in NP_METHODS:
        indexer = np.argsort(indices[valid])
        yvalues[invalid] = np.interp(indices[invalid], indices[valid][indexer], yvalues[valid][indexer])
    else:
        yvalues[invalid] = _interpolate_scipy_wrapper(indices[valid], yvalues[valid], indices[invalid], method=method, fill_value=fill_value, bounds_error=bounds_error, order=order, **kwargs)
    if mask is not None:
        mask[:] = False
        mask[preserve_nans] = True
    elif is_datetimelike:
        yvalues[preserve_nans] = NaT.value
    else:
        yvalues[preserve_nans] = np.nan
    return