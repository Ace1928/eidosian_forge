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
class Weighted(Generic[T_Xarray]):
    """An object that implements weighted operations.

    You should create a Weighted object by using the ``DataArray.weighted`` or
    ``Dataset.weighted`` methods.

    See Also
    --------
    Dataset.weighted
    DataArray.weighted
    """
    __slots__ = ('obj', 'weights')

    def __init__(self, obj: T_Xarray, weights: T_DataArray) -> None:
        """
        Create a Weighted object

        Parameters
        ----------
        obj : DataArray or Dataset
            Object over which the weighted reduction operation is applied.
        weights : DataArray
            An array of weights associated with the values in the obj.
            Each value in the obj contributes to the reduction operation
            according to its associated weight.

        Notes
        -----
        ``weights`` must be a ``DataArray`` and cannot contain missing values.
        Missing values can be replaced by ``weights.fillna(0)``.
        """
        from xarray.core.dataarray import DataArray
        if not isinstance(weights, DataArray):
            raise ValueError('`weights` must be a DataArray')

        def _weight_check(w):
            if duck_array_ops.isnull(w).any():
                raise ValueError('`weights` cannot contain missing values. Missing values can be replaced by `weights.fillna(0)`.')
            return w
        if is_duck_dask_array(weights.data):
            weights = weights.copy(data=weights.data.map_blocks(_weight_check, dtype=weights.dtype), deep=False)
        else:
            _weight_check(weights.data)
        self.obj: T_Xarray = obj
        self.weights: T_DataArray = weights

    def _check_dim(self, dim: Dims):
        """raise an error if any dimension is missing"""
        dims: list[Hashable]
        if isinstance(dim, str) or not isinstance(dim, Iterable):
            dims = [dim] if dim else []
        else:
            dims = list(dim)
        all_dims = set(self.obj.dims).union(set(self.weights.dims))
        missing_dims = set(dims) - all_dims
        if missing_dims:
            raise ValueError(f'Dimensions {tuple(missing_dims)} not found in {self.__class__.__name__} dimensions {tuple(all_dims)}')

    @staticmethod
    def _reduce(da: T_DataArray, weights: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """reduce using dot; equivalent to (da * weights).sum(dim, skipna)

        for internal use only
        """
        if dim is None:
            dim = ...
        if skipna or (skipna is None and da.dtype.kind in 'cfO'):
            da = da.fillna(0.0)
        return dot(da, weights, dim=dim)

    def _sum_of_weights(self, da: T_DataArray, dim: Dims=None) -> T_DataArray:
        """Calculate the sum of weights, accounting for missing values"""
        mask = da.notnull()
        if self.weights.dtype == bool:
            sum_of_weights = self._reduce(mask, duck_array_ops.astype(self.weights, dtype=int), dim=dim, skipna=False)
        else:
            sum_of_weights = self._reduce(mask, self.weights, dim=dim, skipna=False)
        valid_weights = sum_of_weights != 0.0
        return sum_of_weights.where(valid_weights)

    def _sum_of_squares(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``sum_of_squares`` along some dimension(s)."""
        demeaned = da - da.weighted(self.weights).mean(dim=dim)
        return self._reduce(demeaned ** 2, self.weights, dim=dim, skipna=skipna)

    def _weighted_sum(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``sum`` along some dimension(s)."""
        return self._reduce(da, self.weights, dim=dim, skipna=skipna)

    def _weighted_mean(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``mean`` along some dimension(s)."""
        weighted_sum = self._weighted_sum(da, dim=dim, skipna=skipna)
        sum_of_weights = self._sum_of_weights(da, dim=dim)
        return weighted_sum / sum_of_weights

    def _weighted_var(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``var`` along some dimension(s)."""
        sum_of_squares = self._sum_of_squares(da, dim=dim, skipna=skipna)
        sum_of_weights = self._sum_of_weights(da, dim=dim)
        return sum_of_squares / sum_of_weights

    def _weighted_std(self, da: T_DataArray, dim: Dims=None, skipna: bool | None=None) -> T_DataArray:
        """Reduce a DataArray by a weighted ``std`` along some dimension(s)."""
        return cast('T_DataArray', np.sqrt(self._weighted_var(da, dim, skipna)))

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

    def _implementation(self, func, dim, **kwargs):
        raise NotImplementedError('Use `Dataset.weighted` or `DataArray.weighted`')

    @_deprecate_positional_args('v2023.10.0')
    def sum_of_weights(self, dim: Dims=None, *, keep_attrs: bool | None=None) -> T_Xarray:
        return self._implementation(self._sum_of_weights, dim=dim, keep_attrs=keep_attrs)

    @_deprecate_positional_args('v2023.10.0')
    def sum_of_squares(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None) -> T_Xarray:
        return self._implementation(self._sum_of_squares, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    @_deprecate_positional_args('v2023.10.0')
    def sum(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None) -> T_Xarray:
        return self._implementation(self._weighted_sum, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    @_deprecate_positional_args('v2023.10.0')
    def mean(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None) -> T_Xarray:
        return self._implementation(self._weighted_mean, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    @_deprecate_positional_args('v2023.10.0')
    def var(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None) -> T_Xarray:
        return self._implementation(self._weighted_var, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    @_deprecate_positional_args('v2023.10.0')
    def std(self, dim: Dims=None, *, skipna: bool | None=None, keep_attrs: bool | None=None) -> T_Xarray:
        return self._implementation(self._weighted_std, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    def quantile(self, q: ArrayLike, *, dim: Dims=None, keep_attrs: bool | None=None, skipna: bool=True) -> T_Xarray:
        return self._implementation(self._weighted_quantile, q=q, dim=dim, skipna=skipna, keep_attrs=keep_attrs)

    def __repr__(self) -> str:
        """provide a nice str repr of our Weighted object"""
        klass = self.__class__.__name__
        weight_dims = ', '.join(map(str, self.weights.dims))
        return f'{klass} with weights along dimensions: {weight_dims}'