from __future__ import annotations
from typing import (
import numpy as np
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.dtypes.missing import isna
from pandas import (
import pandas.core.algorithms as algos
from pandas.core.arrays.datetimelike import dtype_to_unit
def _bins_to_cuts(x_idx: Index, bins: Index, right: bool=True, labels=None, precision: int=3, include_lowest: bool=False, duplicates: str='raise', ordered: bool=True):
    if not ordered and labels is None:
        raise ValueError("'labels' must be provided if 'ordered = False'")
    if duplicates not in ['raise', 'drop']:
        raise ValueError("invalid value for 'duplicates' parameter, valid options are: raise, drop")
    result: Categorical | np.ndarray
    if isinstance(bins, IntervalIndex):
        ids = bins.get_indexer(x_idx)
        cat_dtype = CategoricalDtype(bins, ordered=True)
        result = Categorical.from_codes(ids, dtype=cat_dtype, validate=False)
        return (result, bins)
    unique_bins = algos.unique(bins)
    if len(unique_bins) < len(bins) and len(bins) != 2:
        if duplicates == 'raise':
            raise ValueError(f"Bin edges must be unique: {repr(bins)}.\nYou can drop duplicate edges by setting the 'duplicates' kwarg")
        bins = unique_bins
    side: Literal['left', 'right'] = 'left' if right else 'right'
    try:
        ids = bins.searchsorted(x_idx, side=side)
    except TypeError as err:
        if x_idx.dtype.kind == 'm':
            raise ValueError('bins must be of timedelta64 dtype') from err
        elif x_idx.dtype.kind == bins.dtype.kind == 'M':
            raise ValueError('Cannot use timezone-naive bins with timezone-aware values, or vice-versa') from err
        elif x_idx.dtype.kind == 'M':
            raise ValueError('bins must be of datetime64 dtype') from err
        else:
            raise
    ids = ensure_platform_int(ids)
    if include_lowest:
        ids[x_idx == bins[0]] = 1
    na_mask = isna(x_idx) | (ids == len(bins)) | (ids == 0)
    has_nas = na_mask.any()
    if labels is not False:
        if not (labels is None or is_list_like(labels)):
            raise ValueError('Bin labels must either be False, None or passed in as a list-like argument')
        if labels is None:
            labels = _format_labels(bins, precision, right=right, include_lowest=include_lowest)
        elif ordered and len(set(labels)) != len(labels):
            raise ValueError('labels must be unique if ordered=True; pass ordered=False for duplicate labels')
        elif len(labels) != len(bins) - 1:
            raise ValueError('Bin labels must be one fewer than the number of bin edges')
        if not isinstance(getattr(labels, 'dtype', None), CategoricalDtype):
            labels = Categorical(labels, categories=labels if len(set(labels)) == len(labels) else None, ordered=ordered)
        np.putmask(ids, na_mask, 0)
        result = algos.take_nd(labels, ids - 1)
    else:
        result = ids - 1
        if has_nas:
            result = result.astype(np.float64)
            np.putmask(result, na_mask, np.nan)
    return (result, bins)