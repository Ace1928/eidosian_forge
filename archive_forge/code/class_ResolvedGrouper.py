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
class ResolvedGrouper(Generic[T_DataWithCoords]):
    """
    Wrapper around a Grouper object.

    The Grouper object represents an abstract instruction to group an object.
    The ResolvedGrouper object is a concrete version that contains all the common
    logic necessary for a GroupBy problem including the intermediates necessary for
    executing a GroupBy calculation. Specialization to the grouping problem at hand,
    is accomplished by calling the `factorize` method on the encapsulated Grouper
    object.

    This class is private API, while Groupers are public.
    """
    grouper: Grouper
    group: T_Group
    obj: T_DataWithCoords
    codes: DataArray = field(init=False)
    full_index: pd.Index = field(init=False)
    group_indices: T_GroupIndices = field(init=False)
    unique_coord: IndexVariable | _DummyGroup = field(init=False)
    group1d: T_Group = field(init=False)
    stacked_obj: T_DataWithCoords = field(init=False)
    stacked_dim: Hashable | None = field(init=False)
    inserted_dims: list[Hashable] = field(init=False)

    def __post_init__(self) -> None:
        self.grouper = copy.deepcopy(self.grouper)
        self.group: T_Group = _resolve_group(self.obj, self.group)
        self.group1d, self.stacked_obj, self.stacked_dim, self.inserted_dims = _ensure_1d(group=self.group, obj=self.obj)
        self.factorize()

    @property
    def name(self) -> Hashable:
        return self.unique_coord.name

    @property
    def size(self) -> int:
        return len(self)

    def __len__(self) -> int:
        return len(self.full_index)

    @property
    def dims(self):
        return self.group1d.dims

    def factorize(self) -> None:
        encoded = self.grouper.factorize(self.group1d)
        self.codes = encoded.codes
        self.full_index = encoded.full_index
        if encoded.group_indices is not None:
            self.group_indices = encoded.group_indices
        else:
            self.group_indices = [g for g in _codes_to_group_indices(self.codes.data, len(self.full_index)) if g]
        if encoded.unique_coord is None:
            unique_values = self.full_index[np.unique(encoded.codes)]
            self.unique_coord = IndexVariable(self.group.name, unique_values, attrs=self.group.attrs)
        else:
            self.unique_coord = encoded.unique_coord