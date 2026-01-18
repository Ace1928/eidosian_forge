from __future__ import annotations
import itertools
from typing import (
import warnings
import numpy as np
import pandas._libs.reshape as libreshape
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.missing import notna
import pandas.core.algorithms as algos
from pandas.core.algorithms import (
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.construction import ensure_wrapped_if_datetimelike
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.series import Series
from pandas.core.sorting import (
class _Unstacker:
    """
    Helper class to unstack data / pivot with multi-level index

    Parameters
    ----------
    index : MultiIndex
    level : int or str, default last level
        Level to "unstack". Accepts a name for the level.
    fill_value : scalar, optional
        Default value to fill in missing values if subgroups do not have the
        same set of labels. By default, missing values will be replaced with
        the default fill value for that data type, NaN for float, NaT for
        datetimelike, etc. For integer types, by default data will converted to
        float and missing values will be set to NaN.
    constructor : object
        Pandas ``DataFrame`` or subclass used to create unstacked
        response.  If None, DataFrame will be used.

    Examples
    --------
    >>> index = pd.MultiIndex.from_tuples([('one', 'a'), ('one', 'b'),
    ...                                    ('two', 'a'), ('two', 'b')])
    >>> s = pd.Series(np.arange(1, 5, dtype=np.int64), index=index)
    >>> s
    one  a    1
         b    2
    two  a    3
         b    4
    dtype: int64

    >>> s.unstack(level=-1)
         a  b
    one  1  2
    two  3  4

    >>> s.unstack(level=0)
       one  two
    a    1    3
    b    2    4

    Returns
    -------
    unstacked : DataFrame
    """

    def __init__(self, index: MultiIndex, level: Level, constructor, sort: bool=True) -> None:
        self.constructor = constructor
        self.sort = sort
        self.index = index.remove_unused_levels()
        self.level = self.index._get_level_number(level)
        self.lift = 1 if -1 in self.index.codes[self.level] else 0
        self.new_index_levels = list(self.index.levels)
        self.new_index_names = list(self.index.names)
        self.removed_name = self.new_index_names.pop(self.level)
        self.removed_level = self.new_index_levels.pop(self.level)
        self.removed_level_full = index.levels[self.level]
        if not self.sort:
            unique_codes = unique(self.index.codes[self.level])
            self.removed_level = self.removed_level.take(unique_codes)
            self.removed_level_full = self.removed_level_full.take(unique_codes)
        num_rows = np.max([index_level.size for index_level in self.new_index_levels])
        num_columns = self.removed_level.size
        num_cells = num_rows * num_columns
        if num_cells > np.iinfo(np.int32).max:
            warnings.warn(f'The following operation may generate {num_cells} cells in the resulting pandas object.', PerformanceWarning, stacklevel=find_stack_level())
        self._make_selectors()

    @cache_readonly
    def _indexer_and_to_sort(self) -> tuple[npt.NDArray[np.intp], list[np.ndarray]]:
        v = self.level
        codes = list(self.index.codes)
        levs = list(self.index.levels)
        to_sort = codes[:v] + codes[v + 1:] + [codes[v]]
        sizes = tuple((len(x) for x in levs[:v] + levs[v + 1:] + [levs[v]]))
        comp_index, obs_ids = get_compressed_ids(to_sort, sizes)
        ngroups = len(obs_ids)
        indexer = get_group_index_sorter(comp_index, ngroups)
        return (indexer, to_sort)

    @cache_readonly
    def sorted_labels(self) -> list[np.ndarray]:
        indexer, to_sort = self._indexer_and_to_sort
        if self.sort:
            return [line.take(indexer) for line in to_sort]
        return to_sort

    def _make_sorted_values(self, values: np.ndarray) -> np.ndarray:
        if self.sort:
            indexer, _ = self._indexer_and_to_sort
            sorted_values = algos.take_nd(values, indexer, axis=0)
            return sorted_values
        return values

    def _make_selectors(self):
        new_levels = self.new_index_levels
        remaining_labels = self.sorted_labels[:-1]
        level_sizes = tuple((len(x) for x in new_levels))
        comp_index, obs_ids = get_compressed_ids(remaining_labels, level_sizes)
        ngroups = len(obs_ids)
        comp_index = ensure_platform_int(comp_index)
        stride = self.index.levshape[self.level] + self.lift
        self.full_shape = (ngroups, stride)
        selector = self.sorted_labels[-1] + stride * comp_index + self.lift
        mask = np.zeros(np.prod(self.full_shape), dtype=bool)
        mask.put(selector, True)
        if mask.sum() < len(self.index):
            raise ValueError('Index contains duplicate entries, cannot reshape')
        self.group_index = comp_index
        self.mask = mask
        if self.sort:
            self.compressor = comp_index.searchsorted(np.arange(ngroups))
        else:
            self.compressor = np.sort(np.unique(comp_index, return_index=True)[1])

    @cache_readonly
    def mask_all(self) -> bool:
        return bool(self.mask.all())

    @cache_readonly
    def arange_result(self) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
        dummy_arr = np.arange(len(self.index), dtype=np.intp)
        new_values, mask = self.get_new_values(dummy_arr, fill_value=-1)
        return (new_values, mask.any(0))

    def get_result(self, values, value_columns, fill_value) -> DataFrame:
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if value_columns is None and values.shape[1] != 1:
            raise ValueError('must pass column labels for multi-column data')
        values, _ = self.get_new_values(values, fill_value)
        columns = self.get_new_columns(value_columns)
        index = self.new_index
        return self.constructor(values, index=index, columns=columns, dtype=values.dtype)

    def get_new_values(self, values, fill_value=None):
        if values.ndim == 1:
            values = values[:, np.newaxis]
        sorted_values = self._make_sorted_values(values)
        length, width = self.full_shape
        stride = values.shape[1]
        result_width = width * stride
        result_shape = (length, result_width)
        mask = self.mask
        mask_all = self.mask_all
        if mask_all and len(values):
            new_values = sorted_values.reshape(length, width, stride).swapaxes(1, 2).reshape(result_shape)
            new_mask = np.ones(result_shape, dtype=bool)
            return (new_values, new_mask)
        dtype = values.dtype
        if mask_all:
            dtype = values.dtype
            new_values = np.empty(result_shape, dtype=dtype)
        elif isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            new_values = cls._empty(result_shape, dtype=dtype)
            new_values[:] = fill_value
        else:
            dtype, fill_value = maybe_promote(dtype, fill_value)
            new_values = np.empty(result_shape, dtype=dtype)
            new_values.fill(fill_value)
        name = dtype.name
        new_mask = np.zeros(result_shape, dtype=bool)
        if needs_i8_conversion(values.dtype):
            sorted_values = sorted_values.view('i8')
            new_values = new_values.view('i8')
        else:
            sorted_values = sorted_values.astype(name, copy=False)
        libreshape.unstack(sorted_values, mask.view('u1'), stride, length, width, new_values, new_mask.view('u1'))
        if needs_i8_conversion(values.dtype):
            new_values = new_values.view('M8[ns]')
            new_values = ensure_wrapped_if_datetimelike(new_values)
            new_values = new_values.view(values.dtype)
        return (new_values, new_mask)

    def get_new_columns(self, value_columns: Index | None):
        if value_columns is None:
            if self.lift == 0:
                return self.removed_level._rename(name=self.removed_name)
            lev = self.removed_level.insert(0, item=self.removed_level._na_value)
            return lev.rename(self.removed_name)
        stride = len(self.removed_level) + self.lift
        width = len(value_columns)
        propagator = np.repeat(np.arange(width), stride)
        new_levels: FrozenList | list[Index]
        if isinstance(value_columns, MultiIndex):
            new_levels = value_columns.levels + (self.removed_level_full,)
            new_names = value_columns.names + (self.removed_name,)
            new_codes = [lab.take(propagator) for lab in value_columns.codes]
        else:
            new_levels = [value_columns, self.removed_level_full]
            new_names = [value_columns.name, self.removed_name]
            new_codes = [propagator]
        repeater = self._repeater
        new_codes.append(np.tile(repeater, width))
        return MultiIndex(levels=new_levels, codes=new_codes, names=new_names, verify_integrity=False)

    @cache_readonly
    def _repeater(self) -> np.ndarray:
        if len(self.removed_level_full) != len(self.removed_level):
            repeater = self.removed_level_full.get_indexer(self.removed_level)
            if self.lift:
                repeater = np.insert(repeater, 0, -1)
        else:
            stride = len(self.removed_level) + self.lift
            repeater = np.arange(stride) - self.lift
        return repeater

    @cache_readonly
    def new_index(self) -> MultiIndex:
        result_codes = [lab.take(self.compressor) for lab in self.sorted_labels[:-1]]
        if len(self.new_index_levels) == 1:
            level, level_codes = (self.new_index_levels[0], result_codes[0])
            if (level_codes == -1).any():
                level = level.insert(len(level), level._na_value)
            return level.take(level_codes).rename(self.new_index_names[0])
        return MultiIndex(levels=self.new_index_levels, codes=result_codes, names=self.new_index_names, verify_integrity=False)