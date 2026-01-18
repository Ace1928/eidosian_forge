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
class DataArrayGroupByBase(GroupBy['DataArray'], DataArrayGroupbyArithmetic):
    """GroupBy object specialized to grouping DataArray objects"""
    __slots__ = ()
    _dims: tuple[Hashable, ...] | None

    @property
    def dims(self) -> tuple[Hashable, ...]:
        if self._dims is None:
            grouper, = self.groupers
            index = _maybe_squeeze_indices(self._group_indices[0], self._squeeze, grouper, warn=True)
            self._dims = self._obj.isel({self._group_dim: index}).dims
        return self._dims

    def _iter_grouped_shortcut(self, warn_squeeze=True):
        """Fast version of `_iter_grouped` that yields Variables without
        metadata
        """
        var = self._obj.variable
        grouper, = self.groupers
        for idx, indices in enumerate(self._group_indices):
            indices = _maybe_squeeze_indices(indices, self._squeeze, grouper, warn=warn_squeeze and idx == 0)
            yield var[{self._group_dim: indices}]

    def _concat_shortcut(self, applied, dim, positions=None):
        stacked = Variable.concat(applied, dim, shortcut=True)
        grouper, = self.groupers
        reordered = _maybe_reorder(stacked, dim, positions, N=grouper.group.size)
        return self._obj._replace_maybe_drop_dims(reordered)

    def _restore_dim_order(self, stacked: DataArray) -> DataArray:
        grouper, = self.groupers
        group = grouper.group1d
        groupby_coord = f'{group.name}_bins' if isinstance(grouper.grouper, BinGrouper) else group.name

        def lookup_order(dimension):
            if dimension == groupby_coord:
                dimension, = group.dims
            if dimension in self._obj.dims:
                axis = self._obj.get_axis_num(dimension)
            else:
                axis = 1000000.0
            return axis
        new_order = sorted(stacked.dims, key=lookup_order)
        return stacked.transpose(*new_order, transpose_coords=self._restore_coord_dims)

    def map(self, func: Callable[..., DataArray], args: tuple[Any, ...]=(), shortcut: bool | None=None, **kwargs: Any) -> DataArray:
        """Apply a function to each array in the group and concatenate them
        together into a new array.

        `func` is called like `func(ar, *args, **kwargs)` for each array `ar`
        in this group.

        Apply uses heuristics (like `pandas.GroupBy.apply`) to figure out how
        to stack together the array. The rule is:

        1. If the dimension along which the group coordinate is defined is
           still in the first grouped array after applying `func`, then stack
           over this dimension.
        2. Otherwise, stack over the new dimension given by name of this
           grouping (the argument to the `groupby` function).

        Parameters
        ----------
        func : callable
            Callable to apply to each array.
        shortcut : bool, optional
            Whether or not to shortcut evaluation under the assumptions that:

            (1) The action of `func` does not depend on any of the array
                metadata (attributes or coordinates) but only on the data and
                dimensions.
            (2) The action of `func` creates arrays with homogeneous metadata,
                that is, with the same dimensions and attributes.

            If these conditions are satisfied `shortcut` provides significant
            speedup. This should be the case for many common groupby operations
            (e.g., applying numpy ufuncs).
        *args : tuple, optional
            Positional arguments passed to `func`.
        **kwargs
            Used to call `func(ar, **kwargs)` for each array `ar`.

        Returns
        -------
        applied : DataArray
            The result of splitting, applying and combining this array.
        """
        return self._map_maybe_warn(func, args, warn_squeeze=True, shortcut=shortcut, **kwargs)

    def _map_maybe_warn(self, func: Callable[..., DataArray], args: tuple[Any, ...]=(), *, warn_squeeze: bool=True, shortcut: bool | None=None, **kwargs: Any) -> DataArray:
        grouped = self._iter_grouped_shortcut(warn_squeeze) if shortcut else self._iter_grouped(warn_squeeze)
        applied = (maybe_wrap_array(arr, func(arr, *args, **kwargs)) for arr in grouped)
        return self._combine(applied, shortcut=shortcut)

    def apply(self, func, shortcut=False, args=(), **kwargs):
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        DataArrayGroupBy.map
        """
        warnings.warn('GroupBy.apply may be deprecated in the future. Using GroupBy.map is encouraged', PendingDeprecationWarning, stacklevel=2)
        return self.map(func, shortcut=shortcut, args=args, **kwargs)

    def _combine(self, applied, shortcut=False):
        """Recombine the applied objects like the original."""
        applied_example, applied = peek_at(applied)
        coord, dim, positions = self._infer_concat_args(applied_example)
        if shortcut:
            combined = self._concat_shortcut(applied, dim, positions)
        else:
            combined = concat(applied, dim)
            grouper, = self.groupers
            combined = _maybe_reorder(combined, dim, positions, N=grouper.group.size)
        if isinstance(combined, type(self._obj)):
            combined = self._restore_dim_order(combined)
        if coord is not None and dim not in applied_example.dims:
            index, index_vars = create_default_index_implicit(coord)
            indexes = {k: index for k in index_vars}
            combined = combined._overwrite_indexes(indexes, index_vars)
        combined = self._maybe_restore_empty_groups(combined)
        combined = self._maybe_unstack(combined)
        return combined

    def reduce(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> DataArray:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. If None, apply over the
            groupby dimension, if "..." apply over all dimensions.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = [self._group_dim]
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        def reduce_array(ar: DataArray) -> DataArray:
            return ar.reduce(func=func, dim=dim, axis=axis, keep_attrs=keep_attrs, keepdims=keepdims, **kwargs)
        check_reduce_dims(dim, self.dims)
        return self.map(reduce_array, shortcut=shortcut)

    def _reduce_without_squeeze_warn(self, func: Callable[..., Any], dim: Dims=None, *, axis: int | Sequence[int] | None=None, keep_attrs: bool | None=None, keepdims: bool=False, shortcut: bool=True, **kwargs: Any) -> DataArray:
        """Reduce the items in this group by applying `func` along some
        dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `func(x, axis=axis, **kwargs)` to return the result of collapsing
            an np.ndarray over an integer valued axis.
        dim : "...", str, Iterable of Hashable or None, optional
            Dimension(s) over which to apply `func`. If None, apply over the
            groupby dimension, if "..." apply over all dimensions.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `func`. Only one of the 'dimension'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `func` is calculated over all dimension for each group item.
        keep_attrs : bool, optional
            If True, the datasets's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : Array
            Array with summarized data and the indicated dimension(s)
            removed.
        """
        if dim is None:
            dim = [self._group_dim]
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        def reduce_array(ar: DataArray) -> DataArray:
            return ar.reduce(func=func, dim=dim, axis=axis, keep_attrs=keep_attrs, keepdims=keepdims, **kwargs)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='The `squeeze` kwarg')
            check_reduce_dims(dim, self.dims)
        return self._map_maybe_warn(reduce_array, shortcut=shortcut, warn_squeeze=False)