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