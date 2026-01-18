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
def _weighted_quantile(self, da: T_DataArray, q: ArrayLike, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
    """Apply a weighted ``quantile`` to a DataArray along some dimension(s)."""

    def _get_h(n: float, q: np.ndarray, method: QUANTILE_METHODS) -> np.ndarray:
        """Return the interpolation parameter."""
        h: np.ndarray
        if method == 'linear':
            h = (n - 1) * q + 1
        elif method == 'interpolated_inverted_cdf':
            h = n * q
        elif method == 'hazen':
            h = n * q + 0.5
        elif method == 'weibull':
            h = (n + 1) * q
        elif method == 'median_unbiased':
            h = (n + 1 / 3) * q + 1 / 3
        elif method == 'normal_unbiased':
            h = (n + 1 / 4) * q + 3 / 8
        else:
            raise ValueError(f'Invalid method: {method}.')
        return h.clip(1, n)

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
    if skipna is None and da.dtype.kind in 'cfO':
        skipna = True
    q = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if q.ndim > 1:
        raise ValueError('q must be a scalar or 1d')
    if np.any((q < 0) | (q > 1)):
        raise ValueError('q values must be between 0 and 1')
    if dim is None:
        dim = da.dims
    if utils.is_scalar(dim):
        dim = [dim]
    dim = cast(Sequence, dim)
    da, weights = align(da, self.weights, join='inner')
    da, weights = broadcast(da, weights)
    result = apply_ufunc(_weighted_quantile_1d, da, weights, input_core_dims=[dim, dim], output_core_dims=[['quantile']], output_dtypes=[np.float64], dask_gufunc_kwargs=dict(output_sizes={'quantile': len(q)}), dask='parallelized', vectorize=True, kwargs={'q': q, 'skipna': skipna})
    result = result.transpose('quantile', ...)
    result = result.assign_coords(quantile=q).squeeze()
    return result