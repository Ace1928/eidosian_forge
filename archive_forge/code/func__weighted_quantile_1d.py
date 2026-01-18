from __future__ import annotations
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Generic, Literal, cast
import numpy as np
from numpy.typing import ArrayLike
from xarray.core import duck_array_ops, utils
from xarray.core.alignment import align, broadcast
from xarray.core.computation import apply_ufunc, dot
from xarray.core.types import Dims, T_DataArray, T_Xarray
from xarray.namedarray.utils import is_duck_dask_array
from xarray.util.deprecation_helpers import _deprecate_positional_args
def _weighted_quantile_1d(data: np.ndarray, weights: np.ndarray, q: np.ndarray, skipna: bool, method: QUANTILE_METHODS='linear') -> np.ndarray:
    is_nan = np.isnan(data)
    if skipna:
        not_nan = ~is_nan
        data = data[not_nan]
        weights = weights[not_nan]
    elif is_nan.any():
        return np.full(q.size, np.nan)
    nonzero_weights = weights != 0
    data = data[nonzero_weights]
    weights = weights[nonzero_weights]
    n = data.size
    if n == 0:
        return np.full(q.size, np.nan)
    nw = weights.sum() ** 2 / (weights ** 2).sum()
    sorter = np.argsort(data)
    data = data[sorter]
    weights = weights[sorter]
    weights = weights / weights.sum()
    weights_cum = np.append(0, weights.cumsum())
    q = np.atleast_2d(q).T
    h = _get_h(nw, q, method)
    u = np.maximum((h - 1) / nw, np.minimum(h / nw, weights_cum))
    v = u * nw - h + 1
    w = np.diff(v)
    return (data * w).sum(axis=1)