from __future__ import annotations
import copy
import datetime
import warnings
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.cftime_offsets import _new_to_legacy_freq
from xarray.core import dtypes, duck_array_ops, nputils, ops
from xarray.core._aggregations import (
from xarray.core.alignment import align
from xarray.core.arithmetic import DataArrayGroupbyArithmetic, DatasetGroupbyArithmetic
from xarray.core.common import ImplementsArrayReduce, ImplementsDatasetReduce
from xarray.core.concat import concat
from xarray.core.formatting import format_array_flat
from xarray.core.indexes import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import (
from xarray.core.utils import (
from xarray.core.variable import IndexVariable, Variable
from xarray.util.deprecation_helpers import _deprecate_positional_args
class GroupBy(Generic[T_Xarray]):
    """A object that implements the split-apply-combine pattern.

    Modeled after `pandas.GroupBy`. The `GroupBy` object can be iterated over
    (unique_value, grouped_array) pairs, but the main way to interact with a
    groupby object are with the `apply` or `reduce` methods. You can also
    directly call numpy methods like `mean` or `std`.

    You should create a GroupBy object by using the `DataArray.groupby` or
    `Dataset.groupby` methods.

    See Also
    --------
    Dataset.groupby
    DataArray.groupby
    """
    __slots__ = ('_full_index', '_inserted_dims', '_group', '_group_dim', '_group_indices', '_groups', 'groupers', '_obj', '_restore_coord_dims', '_stacked_dim', '_unique_coord', '_dims', '_sizes', '_squeeze', '_original_obj', '_original_group', '_bins', '_codes')
    _obj: T_Xarray
    groupers: tuple[ResolvedGrouper]
    _squeeze: bool | None
    _restore_coord_dims: bool
    _original_obj: T_Xarray
    _original_group: T_Group
    _group_indices: T_GroupIndices
    _codes: DataArray
    _group_dim: Hashable
    _groups: dict[GroupKey, GroupIndex] | None
    _dims: tuple[Hashable, ...] | Frozen[Hashable, int] | None
    _sizes: Mapping[Hashable, int] | None

    def __init__(self, obj: T_Xarray, groupers: tuple[ResolvedGrouper], squeeze: bool | None=False, restore_coord_dims: bool=True) -> None:
        """Create a GroupBy object

        Parameters
        ----------
        obj : Dataset or DataArray
            Object to group.
        grouper : Grouper
            Grouper object
        restore_coord_dims : bool, default: True
            If True, also restore the dimension order of multi-dimensional
            coordinates.
        """
        self.groupers = groupers
        self._original_obj = obj
        grouper, = self.groupers
        self._original_group = grouper.group
        self._obj = grouper.stacked_obj
        self._restore_coord_dims = restore_coord_dims
        self._squeeze = squeeze
        self._group_indices = grouper.group_indices
        self._codes = self._maybe_unstack(grouper.codes)
        self._group_dim, = grouper.group1d.dims
        self._groups = None
        self._dims = None
        self._sizes = None

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Ordered mapping from dimension names to lengths.

        Immutable.

        See Also
        --------
        DataArray.sizes
        Dataset.sizes
        """
        if self._sizes is None:
            grouper, = self.groupers
            index = _maybe_squeeze_indices(self._group_indices[0], self._squeeze, grouper, warn=True)
            self._sizes = self._obj.isel({self._group_dim: index}).sizes
        return self._sizes

    def map(self, func: Callable, args: tuple[Any, ...]=(), shortcut: bool | None=None, **kwargs: Any) -> T_Xarray:
        raise NotImplementedError()

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> T_Xarray:
        raise NotImplementedError()

    @property
    def groups(self) -> dict[GroupKey, GroupIndex]:
        """
        Mapping from group labels to indices. The indices can be used to index the underlying object.
        """
        if self._groups is None:
            grouper, = self.groupers
            squeezed_indices = (_maybe_squeeze_indices(ind, self._squeeze, grouper, warn=idx > 0) for idx, ind in enumerate(self._group_indices))
            self._groups = dict(zip(grouper.unique_coord.values, squeezed_indices))
        return self._groups

    def __getitem__(self, key: GroupKey) -> T_Xarray:
        """
        Get DataArray or Dataset corresponding to a particular group label.
        """
        grouper, = self.groupers
        index = _maybe_squeeze_indices(self.groups[key], self._squeeze, grouper, warn=True)
        return self._obj.isel({self._group_dim: index})

    def __len__(self) -> int:
        grouper, = self.groupers
        return grouper.size

    def __iter__(self) -> Iterator[tuple[GroupKey, T_Xarray]]:
        grouper, = self.groupers
        return zip(grouper.unique_coord.data, self._iter_grouped())

    def __repr__(self) -> str:
        grouper, = self.groupers
        return '{}, grouped over {!r}\n{!r} groups with labels {}.'.format(self.__class__.__name__, grouper.name, grouper.full_index.size, ', '.join(format_array_flat(grouper.full_index, 30).split()))

    def _iter_grouped(self, warn_squeeze=True) -> Iterator[T_Xarray]:
        """Iterate over each element in this group"""
        grouper, = self.groupers
        for idx, indices in enumerate(self._group_indices):
            indices = _maybe_squeeze_indices(indices, self._squeeze, grouper, warn=warn_squeeze and idx == 0)
            yield self._obj.isel({self._group_dim: indices})

    def _infer_concat_args(self, applied_example):
        grouper, = self.groupers
        if self._group_dim in applied_example.dims:
            coord = grouper.group1d
            positions = self._group_indices
        else:
            coord = grouper.unique_coord
            positions = None
        dim, = coord.dims
        if isinstance(coord, _DummyGroup):
            coord = None
        coord = getattr(coord, 'variable', coord)
        return (coord, dim, positions)

    def _binary_op(self, other, f, reflexive=False):
        from xarray.core.dataarray import DataArray
        from xarray.core.dataset import Dataset
        g = f if not reflexive else lambda x, y: f(y, x)
        grouper, = self.groupers
        obj = self._original_obj
        group = grouper.group
        codes = self._codes
        dims = group.dims
        if isinstance(group, _DummyGroup):
            group = coord = group.to_dataarray()
        else:
            coord = grouper.unique_coord
            if not isinstance(coord, DataArray):
                coord = DataArray(grouper.unique_coord)
        name = grouper.name
        if not isinstance(other, (Dataset, DataArray)):
            raise TypeError('GroupBy objects only support binary ops when the other argument is a Dataset or DataArray')
        if name not in other.dims:
            raise ValueError(f'incompatible dimensions for a grouped binary operation: the group variable {name!r} is not a dimension on the other argument with dimensions {other.dims!r}')
        for var in other.coords:
            if other[var].ndim == 0:
                other[var] = other[var].drop_vars(var).expand_dims({name: other.sizes[name]})
        mask = codes == -1
        if mask.any():
            obj = obj.where(~mask, drop=True)
            group = group.where(~mask, drop=True)
            codes = codes.where(~mask, drop=True).astype(int)
        if obj.chunks and (not other.chunks):
            chunks = {k: v for k, v in obj.chunksizes.items() if k in other.dims}
            chunks[name] = 1
            other = other.chunk(chunks)
        other, _ = align(other, coord, join='right', copy=False)
        expanded = other.isel({name: codes})
        result = g(obj, expanded)
        if group.ndim > 1:
            for var in set(obj.coords) - set(obj.xindexes):
                if set(obj[var].dims) < set(group.dims):
                    result[var] = obj[var].reset_coords(drop=True).broadcast_like(group)
        if isinstance(result, Dataset) and isinstance(obj, Dataset):
            for var in set(result):
                for d in dims:
                    if d not in obj[var].dims:
                        result[var] = result[var].transpose(d, ...)
        return result

    def _restore_dim_order(self, stacked):
        raise NotImplementedError

    def _maybe_restore_empty_groups(self, combined):
        """Our index contained empty groups (e.g., from a resampling or binning). If we
        reduced on that dimension, we want to restore the full index.
        """
        grouper, = self.groupers
        if isinstance(grouper.grouper, (BinGrouper, TimeResampler)) and grouper.name in combined.dims:
            indexers = {grouper.name: grouper.full_index}
            combined = combined.reindex(**indexers)
        return combined

    def _maybe_unstack(self, obj):
        """This gets called if we are applying on an array with a
        multidimensional group."""
        grouper, = self.groupers
        stacked_dim = grouper.stacked_dim
        inserted_dims = grouper.inserted_dims
        if stacked_dim is not None and stacked_dim in obj.dims:
            obj = obj.unstack(stacked_dim)
            for dim in inserted_dims:
                if dim in obj.coords:
                    del obj.coords[dim]
            obj._indexes = filter_indexes_from_coords(obj._indexes, set(obj.coords))
        return obj

    def _flox_reduce(self, dim: Dims, keep_attrs: bool | None=None, **kwargs: Any):
        """Adaptor function that translates our groupby API to that of flox."""
        import flox
        from flox.xarray import xarray_reduce
        from xarray.core.dataset import Dataset
        obj = self._original_obj
        grouper, = self.groupers
        isbin = isinstance(grouper.grouper, BinGrouper)
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        if Version(flox.__version__) < Version('0.9'):
            kwargs.setdefault('method', 'cohorts')
        numeric_only = kwargs.pop('numeric_only', None)
        if numeric_only:
            non_numeric = {name: var for name, var in obj.data_vars.items() if not (np.issubdtype(var.dtype, np.number) or var.dtype == np.bool_)}
        else:
            non_numeric = {}
        if 'min_count' in kwargs:
            if kwargs['func'] not in ['sum', 'prod']:
                raise TypeError("Received an unexpected keyword argument 'min_count'")
            elif kwargs['min_count'] is None:
                kwargs['min_count'] = 0
        if (dim is None or dim == grouper.name) and grouper.name in obj.xindexes:
            index = obj.indexes[grouper.name]
            if index.is_unique and self._squeeze:
                raise ValueError(f'cannot reduce over dimensions {grouper.name!r}')
        unindexed_dims: tuple[Hashable, ...] = tuple()
        if isinstance(grouper.group, _DummyGroup) and (not isbin):
            unindexed_dims = (grouper.name,)
        parsed_dim: tuple[Hashable, ...]
        if isinstance(dim, str):
            parsed_dim = (dim,)
        elif dim is None:
            parsed_dim = grouper.group.dims
        elif dim is ...:
            parsed_dim = tuple(obj.dims)
        else:
            parsed_dim = tuple(dim)
        if any((d not in grouper.group.dims and d not in obj.dims for d in parsed_dim)):
            raise ValueError(f'cannot reduce over dimensions {dim}.')
        if kwargs['func'] not in ['all', 'any', 'count']:
            kwargs.setdefault('fill_value', np.nan)
        if isbin and kwargs['func'] == 'count':
            kwargs.setdefault('fill_value', np.nan)
            kwargs.setdefault('min_count', 1)
        output_index = grouper.full_index
        result = xarray_reduce(obj.drop_vars(non_numeric.keys()), self._codes, dim=parsed_dim, expected_groups=(pd.RangeIndex(len(output_index)),), isbin=False, keep_attrs=keep_attrs, **kwargs)
        group_dims = grouper.group.dims
        if set(group_dims).issubset(set(parsed_dim)):
            result[grouper.name] = output_index
            result = result.drop_vars(unindexed_dims)
        for name, var in non_numeric.items():
            if all((d not in var.dims for d in parsed_dim)):
                result[name] = var.variable.set_dims((grouper.name,) + var.dims, (result.sizes[grouper.name],) + var.shape)
        if not isinstance(result, Dataset):
            result = self._restore_dim_order(result)
        return result

    def fillna(self, value: Any) -> T_Xarray:
        """Fill missing values in this object by group.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value
            Used to fill all matching missing values by group. Needs
            to be of a valid type for the wrapped object's fillna
            method.

        Returns
        -------
        same type as the grouped object

        See Also
        --------
        Dataset.fillna
        DataArray.fillna
        """
        return ops.fillna(self, value)

    @_deprecate_positional_args('v2023.10.0')
    def quantile(self, q: ArrayLike, dim: Dims=None, *, method: QuantileMethods='linear', keep_attrs: bool | None=None, skipna: bool | None=None, interpolation: QuantileMethods | None=None) -> T_Xarray:
        """Compute the qth quantile over each array in the groups and
        concatenate them together into a new array.

        Parameters
        ----------
        q : float or sequence of float
            Quantile to compute, which must be between 0 and 1
            inclusive.
        dim : str or Iterable of Hashable, optional
            Dimension(s) over which to apply quantile.
            Defaults to the grouped dimension.
        method : str, default: "linear"
            This optional parameter specifies the interpolation method to use when the
            desired quantile lies between two data points. The options sorted by their R
            type as summarized in the H&F paper [1]_ are:

                1. "inverted_cdf"
                2. "averaged_inverted_cdf"
                3. "closest_observation"
                4. "interpolated_inverted_cdf"
                5. "hazen"
                6. "weibull"
                7. "linear"  (default)
                8. "median_unbiased"
                9. "normal_unbiased"

            The first three methods are discontiuous.  The following discontinuous
            variations of the default "linear" (7.) option are also available:

                * "lower"
                * "higher"
                * "midpoint"
                * "nearest"

            See :py:func:`numpy.quantile` or [1]_ for details. The "method" argument
            was previously called "interpolation", renamed in accordance with numpy
            version 1.22.0.
        keep_attrs : bool or None, default: None
            If True, the dataarray's attributes (`attrs`) will be copied from
            the original object to the new one.  If False, the new
            object will be returned without attributes.
        skipna : bool or None, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        quantiles : Variable
            If `q` is a single quantile, then the result is a
            scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile. In either case a
            quantile dimension is added to the return array. The other
            dimensions are the dimensions that remain after the
            reduction of the array.

        See Also
        --------
        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, Dataset.quantile
        DataArray.quantile

        Examples
        --------
        >>> da = xr.DataArray(
        ...     [[1.3, 8.4, 0.7, 6.9], [0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]],
        ...     coords={"x": [0, 0, 1], "y": [1, 1, 2, 2]},
        ...     dims=("x", "y"),
        ... )
        >>> ds = xr.Dataset({"a": da})
        >>> da.groupby("x").quantile(0)
        <xarray.DataArray (x: 2, y: 4)> Size: 64B
        array([[0.7, 4.2, 0.7, 1.5],
               [6.5, 7.3, 2.6, 1.9]])
        Coordinates:
          * y         (y) int64 32B 1 1 2 2
            quantile  float64 8B 0.0
          * x         (x) int64 16B 0 1
        >>> ds.groupby("y").quantile(0, dim=...)
        <xarray.Dataset> Size: 40B
        Dimensions:   (y: 2)
        Coordinates:
            quantile  float64 8B 0.0
          * y         (y) int64 16B 1 2
        Data variables:
            a         (y) float64 16B 0.7 0.7
        >>> da.groupby("x").quantile([0, 0.5, 1])
        <xarray.DataArray (x: 2, y: 4, quantile: 3)> Size: 192B
        array([[[0.7 , 1.  , 1.3 ],
                [4.2 , 6.3 , 8.4 ],
                [0.7 , 5.05, 9.4 ],
                [1.5 , 4.2 , 6.9 ]],
        <BLANKLINE>
               [[6.5 , 6.5 , 6.5 ],
                [7.3 , 7.3 , 7.3 ],
                [2.6 , 2.6 , 2.6 ],
                [1.9 , 1.9 , 1.9 ]]])
        Coordinates:
          * y         (y) int64 32B 1 1 2 2
          * quantile  (quantile) float64 24B 0.0 0.5 1.0
          * x         (x) int64 16B 0 1
        >>> ds.groupby("y").quantile([0, 0.5, 1], dim=...)
        <xarray.Dataset> Size: 88B
        Dimensions:   (y: 2, quantile: 3)
        Coordinates:
          * quantile  (quantile) float64 24B 0.0 0.5 1.0
          * y         (y) int64 16B 1 2
        Data variables:
            a         (y, quantile) float64 48B 0.7 5.35 8.4 0.7 2.25 9.4

        References
        ----------
        .. [1] R. J. Hyndman and Y. Fan,
           "Sample quantiles in statistical packages,"
           The American Statistician, 50(4), pp. 361-365, 1996
        """
        if dim is None:
            grouper, = self.groupers
            dim = grouper.group1d.dims
        q = np.asarray(q, dtype=np.float64)
        if method == 'linear' and OPTIONS['use_flox'] and contains_only_chunked_or_numpy(self._obj) and module_available('flox', minversion='0.9.4'):
            result = self._flox_reduce(func='quantile', q=q, dim=dim, keep_attrs=keep_attrs, skipna=skipna)
            return result
        else:
            return self.map(self._obj.__class__.quantile, shortcut=False, q=q, dim=dim, method=method, keep_attrs=keep_attrs, skipna=skipna, interpolation=interpolation)

    def where(self, cond, other=dtypes.NA) -> T_Xarray:
        """Return elements from `self` or `other` depending on `cond`.

        Parameters
        ----------
        cond : DataArray or Dataset
            Locations at which to preserve this objects values. dtypes have to be `bool`
        other : scalar, DataArray or Dataset, optional
            Value to use for locations in this object where ``cond`` is False.
            By default, inserts missing values.

        Returns
        -------
        same type as the grouped object

        See Also
        --------
        Dataset.where
        """
        return ops.where_method(self, cond, other)

    def _first_or_last(self, op, skipna, keep_attrs):
        if all((isinstance(maybe_slice, slice) and maybe_slice.stop == maybe_slice.start + 1 for maybe_slice in self._group_indices)):
            return self._obj
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        return self.reduce(op, dim=[self._group_dim], skipna=skipna, keep_attrs=keep_attrs)

    def first(self, skipna: bool | None=None, keep_attrs: bool | None=None):
        """Return the first element of each group along the group dimension"""
        return self._first_or_last(duck_array_ops.first, skipna, keep_attrs)

    def last(self, skipna: bool | None=None, keep_attrs: bool | None=None):
        """Return the last element of each group along the group dimension"""
        return self._first_or_last(duck_array_ops.last, skipna, keep_attrs)

    def assign_coords(self, coords=None, **coords_kwargs):
        """Assign coordinates by group.

        See Also
        --------
        Dataset.assign_coords
        Dataset.swap_dims
        """
        coords_kwargs = either_dict_or_kwargs(coords, coords_kwargs, 'assign_coords')
        return self.map(lambda ds: ds.assign_coords(**coords_kwargs))