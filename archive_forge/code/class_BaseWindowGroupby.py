from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
import numpy as np
from pandas._libs.tslibs import (
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
from pandas.core.window.common import (
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.arrays.datetimelike import dtype_to_unit
class BaseWindowGroupby(BaseWindow):
    """
    Provide the groupby windowing facilities.
    """
    _grouper: BaseGrouper
    _as_index: bool
    _attributes: list[str] = ['_grouper']

    def __init__(self, obj: DataFrame | Series, *args, _grouper: BaseGrouper, _as_index: bool=True, **kwargs) -> None:
        from pandas.core.groupby.ops import BaseGrouper
        if not isinstance(_grouper, BaseGrouper):
            raise ValueError('Must pass a BaseGrouper object.')
        self._grouper = _grouper
        self._as_index = _as_index
        obj = obj.drop(columns=self._grouper.names, errors='ignore')
        if kwargs.get('step') is not None:
            raise NotImplementedError('step not implemented for groupby')
        super().__init__(obj, *args, **kwargs)

    def _apply(self, func: Callable[..., Any], name: str, numeric_only: bool=False, numba_args: tuple[Any, ...]=(), **kwargs) -> DataFrame | Series:
        result = super()._apply(func, name, numeric_only, numba_args, **kwargs)
        grouped_object_index = self.obj.index
        grouped_index_name = [*grouped_object_index.names]
        groupby_keys = copy.copy(self._grouper.names)
        result_index_names = groupby_keys + grouped_index_name
        drop_columns = [key for key in self._grouper.names if key not in self.obj.index.names or key is None]
        if len(drop_columns) != len(groupby_keys):
            result = result.drop(columns=drop_columns, errors='ignore')
        codes = self._grouper.codes
        levels = copy.copy(self._grouper.levels)
        group_indices = self._grouper.indices.values()
        if group_indices:
            indexer = np.concatenate(list(group_indices))
        else:
            indexer = np.array([], dtype=np.intp)
        codes = [c.take(indexer) for c in codes]
        if grouped_object_index is not None:
            idx = grouped_object_index.take(indexer)
            if not isinstance(idx, MultiIndex):
                idx = MultiIndex.from_arrays([idx])
            codes.extend(list(idx.codes))
            levels.extend(list(idx.levels))
        result_index = MultiIndex(levels, codes, names=result_index_names, verify_integrity=False)
        result.index = result_index
        if not self._as_index:
            result = result.reset_index(level=list(range(len(groupby_keys))))
        return result

    def _apply_pairwise(self, target: DataFrame | Series, other: DataFrame | Series | None, pairwise: bool | None, func: Callable[[DataFrame | Series, DataFrame | Series], DataFrame | Series], numeric_only: bool) -> DataFrame | Series:
        """
        Apply the given pairwise function given 2 pandas objects (DataFrame/Series)
        """
        target = target.drop(columns=self._grouper.names, errors='ignore')
        result = super()._apply_pairwise(target, other, pairwise, func, numeric_only)
        if other is not None and (not all((len(group) == len(other) for group in self._grouper.indices.values()))):
            old_result_len = len(result)
            result = concat([result.take(gb_indices).reindex(result.index) for gb_indices in self._grouper.indices.values()])
            gb_pairs = (com.maybe_make_list(pair) for pair in self._grouper.indices.keys())
            groupby_codes = []
            groupby_levels = []
            for gb_level_pair in map(list, zip(*gb_pairs)):
                labels = np.repeat(np.array(gb_level_pair), old_result_len)
                codes, levels = factorize(labels)
                groupby_codes.append(codes)
                groupby_levels.append(levels)
        else:
            groupby_codes = self._grouper.codes
            groupby_levels = self._grouper.levels
            group_indices = self._grouper.indices.values()
            if group_indices:
                indexer = np.concatenate(list(group_indices))
            else:
                indexer = np.array([], dtype=np.intp)
            if target.ndim == 1:
                repeat_by = 1
            else:
                repeat_by = len(target.columns)
            groupby_codes = [np.repeat(c.take(indexer), repeat_by) for c in groupby_codes]
        if isinstance(result.index, MultiIndex):
            result_codes = list(result.index.codes)
            result_levels = list(result.index.levels)
            result_names = list(result.index.names)
        else:
            idx_codes, idx_levels = factorize(result.index)
            result_codes = [idx_codes]
            result_levels = [idx_levels]
            result_names = [result.index.name]
        result_codes = groupby_codes + result_codes
        result_levels = groupby_levels + result_levels
        result_names = self._grouper.names + result_names
        result_index = MultiIndex(result_levels, result_codes, names=result_names, verify_integrity=False)
        result.index = result_index
        return result

    def _create_data(self, obj: NDFrameT, numeric_only: bool=False) -> NDFrameT:
        """
        Split data into blocks & return conformed data.
        """
        if not obj.empty:
            groupby_order = np.concatenate(list(self._grouper.indices.values())).astype(np.int64)
            obj = obj.take(groupby_order)
        return super()._create_data(obj, numeric_only)

    def _gotitem(self, key, ndim, subset=None):
        if self.on is not None:
            subset = self.obj.set_index(self._on)
        return super()._gotitem(key, ndim, subset=subset)