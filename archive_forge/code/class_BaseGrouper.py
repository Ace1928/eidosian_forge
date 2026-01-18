from __future__ import annotations
import collections
import functools
from typing import (
import numpy as np
from pandas._libs import (
import pandas._libs.groupby as libgroupby
from pandas._typing import (
from pandas.errors import AbstractMethodError
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
from pandas.core.frame import DataFrame
from pandas.core.groupby import grouper
from pandas.core.indexes.api import (
from pandas.core.series import Series
from pandas.core.sorting import (
class BaseGrouper:
    """
    This is an internal Grouper class, which actually holds
    the generated groups

    Parameters
    ----------
    axis : Index
    groupings : Sequence[Grouping]
        all the grouping instances to handle in this grouper
        for example for grouper list to groupby, need to pass the list
    sort : bool, default True
        whether this grouper will give sorted result or not

    """
    axis: Index

    def __init__(self, axis: Index, groupings: Sequence[grouper.Grouping], sort: bool=True, dropna: bool=True) -> None:
        assert isinstance(axis, Index), axis
        self.axis = axis
        self._groupings: list[grouper.Grouping] = list(groupings)
        self._sort = sort
        self.dropna = dropna

    @property
    def groupings(self) -> list[grouper.Grouping]:
        return self._groupings

    @property
    def shape(self) -> Shape:
        return tuple((ping.ngroups for ping in self.groupings))

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.indices)

    @property
    def nkeys(self) -> int:
        return len(self.groupings)

    def get_iterator(self, data: NDFrameT, axis: AxisInt=0) -> Iterator[tuple[Hashable, NDFrameT]]:
        """
        Groupby iterator

        Returns
        -------
        Generator yielding sequence of (name, subsetted object)
        for each group
        """
        splitter = self._get_splitter(data, axis=axis)
        keys = self.group_keys_seq
        yield from zip(keys, splitter)

    @final
    def _get_splitter(self, data: NDFrame, axis: AxisInt=0) -> DataSplitter:
        """
        Returns
        -------
        Generator yielding subsetted objects
        """
        ids, _, ngroups = self.group_info
        return _get_splitter(data, ids, ngroups, sorted_ids=self._sorted_ids, sort_idx=self._sort_idx, axis=axis)

    @final
    @cache_readonly
    def group_keys_seq(self):
        if len(self.groupings) == 1:
            return self.levels[0]
        else:
            ids, _, ngroups = self.group_info
            return get_flattened_list(ids, ngroups, self.levels, self.codes)

    @cache_readonly
    def indices(self) -> dict[Hashable, npt.NDArray[np.intp]]:
        """dict {group name -> group indices}"""
        if len(self.groupings) == 1 and isinstance(self.result_index, CategoricalIndex):
            return self.groupings[0].indices
        codes_list = [ping.codes for ping in self.groupings]
        keys = [ping._group_index for ping in self.groupings]
        return get_indexer_dict(codes_list, keys)

    @final
    def result_ilocs(self) -> npt.NDArray[np.intp]:
        """
        Get the original integer locations of result_index in the input.
        """
        group_index = get_group_index(self.codes, self.shape, sort=self._sort, xnull=True)
        group_index, _ = compress_group_index(group_index, sort=self._sort)
        if self.has_dropped_na:
            mask = np.where(group_index >= 0)
            null_gaps = np.cumsum(group_index == -1)[mask]
            group_index = group_index[mask]
        result = get_group_index_sorter(group_index, self.ngroups)
        if self.has_dropped_na:
            result += np.take(null_gaps, result)
        return result

    @final
    @property
    def codes(self) -> list[npt.NDArray[np.signedinteger]]:
        return [ping.codes for ping in self.groupings]

    @property
    def levels(self) -> list[Index]:
        return [ping._group_index for ping in self.groupings]

    @property
    def names(self) -> list[Hashable]:
        return [ping.name for ping in self.groupings]

    @final
    def size(self) -> Series:
        """
        Compute group sizes.
        """
        ids, _, ngroups = self.group_info
        out: np.ndarray | list
        if ngroups:
            out = np.bincount(ids[ids != -1], minlength=ngroups)
        else:
            out = []
        return Series(out, index=self.result_index, dtype='int64', copy=False)

    @cache_readonly
    def groups(self) -> dict[Hashable, np.ndarray]:
        """dict {group name -> group labels}"""
        if len(self.groupings) == 1:
            return self.groupings[0].groups
        else:
            to_groupby = []
            for ping in self.groupings:
                gv = ping.grouping_vector
                if not isinstance(gv, BaseGrouper):
                    to_groupby.append(gv)
                else:
                    to_groupby.append(gv.groupings[0].grouping_vector)
            index = MultiIndex.from_arrays(to_groupby)
            return self.axis.groupby(index)

    @final
    @cache_readonly
    def is_monotonic(self) -> bool:
        return Index(self.group_info[0]).is_monotonic_increasing

    @final
    @cache_readonly
    def has_dropped_na(self) -> bool:
        """
        Whether grouper has null value(s) that are dropped.
        """
        return bool((self.group_info[0] < 0).any())

    @cache_readonly
    def group_info(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
        comp_ids, obs_group_ids = self._get_compressed_codes()
        ngroups = len(obs_group_ids)
        comp_ids = ensure_platform_int(comp_ids)
        return (comp_ids, obs_group_ids, ngroups)

    @cache_readonly
    def codes_info(self) -> npt.NDArray[np.intp]:
        ids, _, _ = self.group_info
        return ids

    @final
    def _get_compressed_codes(self) -> tuple[npt.NDArray[np.signedinteger], npt.NDArray[np.intp]]:
        if len(self.groupings) > 1:
            group_index = get_group_index(self.codes, self.shape, sort=True, xnull=True)
            return compress_group_index(group_index, sort=self._sort)
        ping = self.groupings[0]
        return (ping.codes, np.arange(len(ping._group_index), dtype=np.intp))

    @final
    @cache_readonly
    def ngroups(self) -> int:
        return len(self.result_index)

    @property
    def reconstructed_codes(self) -> list[npt.NDArray[np.intp]]:
        codes = self.codes
        ids, obs_ids, _ = self.group_info
        return decons_obs_group_ids(ids, obs_ids, self.shape, codes, xnull=True)

    @cache_readonly
    def result_index(self) -> Index:
        if len(self.groupings) == 1:
            return self.groupings[0]._result_index.rename(self.names[0])
        codes = self.reconstructed_codes
        levels = [ping._result_index for ping in self.groupings]
        return MultiIndex(levels=levels, codes=codes, verify_integrity=False, names=self.names)

    @final
    def get_group_levels(self) -> list[ArrayLike]:
        if len(self.groupings) == 1:
            return [self.groupings[0]._group_arraylike]
        name_list = []
        for ping, codes in zip(self.groupings, self.reconstructed_codes):
            codes = ensure_platform_int(codes)
            levels = ping._group_arraylike.take(codes)
            name_list.append(levels)
        return name_list

    @final
    def _cython_operation(self, kind: str, values, how: str, axis: AxisInt, min_count: int=-1, **kwargs) -> ArrayLike:
        """
        Returns the values of a cython operation.
        """
        assert kind in ['transform', 'aggregate']
        cy_op = WrappedCythonOp(kind=kind, how=how, has_dropped_na=self.has_dropped_na)
        ids, _, _ = self.group_info
        ngroups = self.ngroups
        return cy_op.cython_operation(values=values, axis=axis, min_count=min_count, comp_ids=ids, ngroups=ngroups, **kwargs)

    @final
    def agg_series(self, obj: Series, func: Callable, preserve_dtype: bool=False) -> ArrayLike:
        """
        Parameters
        ----------
        obj : Series
        func : function taking a Series and returning a scalar-like
        preserve_dtype : bool
            Whether the aggregation is known to be dtype-preserving.

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if not isinstance(obj._values, np.ndarray):
            preserve_dtype = True
        result = self._aggregate_series_pure_python(obj, func)
        npvalues = lib.maybe_convert_objects(result, try_float=False)
        if preserve_dtype:
            out = maybe_cast_pointwise_result(npvalues, obj.dtype, numeric_only=True)
        else:
            out = npvalues
        return out

    @final
    def _aggregate_series_pure_python(self, obj: Series, func: Callable) -> npt.NDArray[np.object_]:
        _, _, ngroups = self.group_info
        result = np.empty(ngroups, dtype='O')
        initialized = False
        splitter = self._get_splitter(obj, axis=0)
        for i, group in enumerate(splitter):
            res = func(group)
            res = extract_result(res)
            if not initialized:
                check_result_array(res, group.dtype)
                initialized = True
            result[i] = res
        return result

    @final
    def apply_groupwise(self, f: Callable, data: DataFrame | Series, axis: AxisInt=0) -> tuple[list, bool]:
        mutated = False
        splitter = self._get_splitter(data, axis=axis)
        group_keys = self.group_keys_seq
        result_values = []
        zipped = zip(group_keys, splitter)
        for key, group in zipped:
            object.__setattr__(group, 'name', key)
            group_axes = group.axes
            res = f(group)
            if not mutated and (not _is_indexed_like(res, group_axes, axis)):
                mutated = True
            result_values.append(res)
        if len(group_keys) == 0 and getattr(f, '__name__', None) in ['skew', 'sum', 'prod']:
            f(data.iloc[:0])
        return (result_values, mutated)

    @final
    @cache_readonly
    def _sort_idx(self) -> npt.NDArray[np.intp]:
        ids, _, ngroups = self.group_info
        return get_group_index_sorter(ids, ngroups)

    @final
    @cache_readonly
    def _sorted_ids(self) -> npt.NDArray[np.intp]:
        ids, _, _ = self.group_info
        return ids.take(self._sort_idx)