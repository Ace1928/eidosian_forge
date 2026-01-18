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
class UniqueGrouper(Grouper):
    """Grouper object for grouping by a categorical variable."""
    _group_as_index: pd.Index | None = None

    @property
    def is_unique_and_monotonic(self) -> bool:
        if isinstance(self.group, _DummyGroup):
            return True
        index = self.group_as_index
        return index.is_unique and index.is_monotonic_increasing

    @property
    def group_as_index(self) -> pd.Index:
        if self._group_as_index is None:
            self._group_as_index = self.group.to_index()
        return self._group_as_index

    @property
    def can_squeeze(self) -> bool:
        is_dimension = self.group.dims == (self.group.name,)
        return is_dimension and self.is_unique_and_monotonic

    def factorize(self, group1d) -> EncodedGroups:
        self.group = group1d
        if self.can_squeeze:
            return self._factorize_dummy()
        else:
            return self._factorize_unique()

    def _factorize_unique(self) -> EncodedGroups:
        sort = not isinstance(self.group_as_index, pd.MultiIndex)
        unique_values, codes_ = unique_value_groups(self.group_as_index, sort=sort)
        if (codes_ == -1).all():
            raise ValueError('Failed to group data. Are you grouping by a variable that is all NaN?')
        codes = self.group.copy(data=codes_)
        unique_coord = IndexVariable(self.group.name, unique_values, attrs=self.group.attrs)
        full_index = unique_coord
        return EncodedGroups(codes=codes, full_index=full_index, unique_coord=unique_coord)

    def _factorize_dummy(self) -> EncodedGroups:
        size = self.group.size
        group_indices: T_GroupIndices = [slice(i, i + 1) for i in range(size)]
        size_range = np.arange(size)
        if isinstance(self.group, _DummyGroup):
            codes = self.group.to_dataarray().copy(data=size_range)
        else:
            codes = self.group.copy(data=size_range)
        unique_coord = self.group
        full_index = IndexVariable(self.group.name, unique_coord.values, self.group.attrs)
        return EncodedGroups(codes=codes, group_indices=group_indices, full_index=full_index, unique_coord=unique_coord)