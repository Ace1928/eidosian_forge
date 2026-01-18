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
def _factorize_unique(self) -> EncodedGroups:
    sort = not isinstance(self.group_as_index, pd.MultiIndex)
    unique_values, codes_ = unique_value_groups(self.group_as_index, sort=sort)
    if (codes_ == -1).all():
        raise ValueError('Failed to group data. Are you grouping by a variable that is all NaN?')
    codes = self.group.copy(data=codes_)
    unique_coord = IndexVariable(self.group.name, unique_values, attrs=self.group.attrs)
    full_index = unique_coord
    return EncodedGroups(codes=codes, full_index=full_index, unique_coord=unique_coord)