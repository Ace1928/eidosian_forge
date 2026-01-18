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
@dataclass
class TimeResampler(Resampler):
    """Grouper object specialized to resampling the time coordinate."""
    freq: str
    closed: SideOptions | None = field(default=None)
    label: SideOptions | None = field(default=None)
    origin: str | DatetimeLike = field(default='start_day')
    offset: pd.Timedelta | datetime.timedelta | str | None = field(default=None)
    loffset: datetime.timedelta | str | None = field(default=None)
    base: int | None = field(default=None)
    index_grouper: CFTimeGrouper | pd.Grouper = field(init=False)
    group_as_index: pd.Index = field(init=False)

    def __post_init__(self):
        if self.loffset is not None:
            emit_user_level_warning('Following pandas, the `loffset` parameter to resample is deprecated.  Switch to updating the resampled dataset time coordinate using time offset arithmetic.  For example:\n    >>> offset = pd.tseries.frequencies.to_offset(freq) / 2\n    >>> resampled_ds["time"] = resampled_ds.get_index("time") + offset', FutureWarning)
        if self.base is not None:
            emit_user_level_warning('Following pandas, the `base` parameter to resample will be deprecated in a future version of xarray.  Switch to using `origin` or `offset` instead.', FutureWarning)
        if self.base is not None and self.offset is not None:
            raise ValueError('base and offset cannot be present at the same time')

    def _init_properties(self, group: T_Group) -> None:
        from xarray import CFTimeIndex
        from xarray.core.pdcompat import _convert_base_to_offset
        group_as_index = safe_cast_to_index(group)
        if self.base is not None:
            offset = _convert_base_to_offset(self.base, self.freq, group_as_index)
        else:
            offset = self.offset
        if not group_as_index.is_monotonic_increasing:
            raise ValueError('index must be monotonic for resampling')
        if isinstance(group_as_index, CFTimeIndex):
            from xarray.core.resample_cftime import CFTimeGrouper
            index_grouper = CFTimeGrouper(freq=self.freq, closed=self.closed, label=self.label, origin=self.origin, offset=offset, loffset=self.loffset)
        else:
            index_grouper = pd.Grouper(freq=_new_to_legacy_freq(self.freq), closed=self.closed, label=self.label, origin=self.origin, offset=offset)
        self.index_grouper = index_grouper
        self.group_as_index = group_as_index

    def _get_index_and_items(self) -> tuple[pd.Index, pd.Series, np.ndarray]:
        first_items, codes = self.first_items()
        full_index = first_items.index
        if first_items.isnull().any():
            first_items = first_items.dropna()
        full_index = full_index.rename('__resample_dim__')
        return (full_index, first_items, codes)

    def first_items(self) -> tuple[pd.Series, np.ndarray]:
        from xarray import CFTimeIndex
        if isinstance(self.group_as_index, CFTimeIndex):
            return self.index_grouper.first_items(self.group_as_index)
        else:
            s = pd.Series(np.arange(self.group_as_index.size), self.group_as_index)
            grouped = s.groupby(self.index_grouper)
            first_items = grouped.first()
            counts = grouped.count()
            codes = np.repeat(np.arange(len(first_items)), counts)
            if self.loffset is not None:
                _apply_loffset(self.loffset, first_items)
            return (first_items, codes)

    def factorize(self, group) -> EncodedGroups:
        self._init_properties(group)
        full_index, first_items, codes_ = self._get_index_and_items()
        sbins = first_items.values.astype(np.int64)
        group_indices: T_GroupIndices = [slice(i, j) for i, j in zip(sbins[:-1], sbins[1:])]
        group_indices += [slice(sbins[-1], None)]
        unique_coord = IndexVariable(group.name, first_items.index, group.attrs)
        codes = group.copy(data=codes_)
        return EncodedGroups(codes=codes, group_indices=group_indices, full_index=full_index, unique_coord=unique_coord)