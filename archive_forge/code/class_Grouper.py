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
class Grouper(ABC):
    """Base class for Grouper objects that allow specializing GroupBy instructions."""

    @property
    def can_squeeze(self) -> bool:
        """TODO: delete this when the `squeeze` kwarg is deprecated. Only `UniqueGrouper`
        should override it."""
        return False

    @abstractmethod
    def factorize(self, group) -> EncodedGroups:
        """
        Takes the group, and creates intermediates necessary for GroupBy.
        These intermediates are
        1. codes - Same shape as `group` containing a unique integer code for each group.
        2. group_indices - Indexes that let us index out the members of each group.
        3. unique_coord - Unique groups present in the dataset.
        4. full_index - Unique groups in the output. This differs from `unique_coord` in the
           case of resampling and binning, where certain groups in the output are not present in
           the input.
        """
        pass