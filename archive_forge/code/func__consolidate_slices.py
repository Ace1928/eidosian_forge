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
def _consolidate_slices(slices: list[slice]) -> list[slice]:
    """Consolidate adjacent slices in a list of slices."""
    result: list[slice] = []
    last_slice = slice(None)
    for slice_ in slices:
        if not isinstance(slice_, slice):
            raise ValueError(f'list element is not a slice: {slice_!r}')
        if result and last_slice.stop == slice_.start and _is_one_or_none(last_slice.step) and _is_one_or_none(slice_.step):
            last_slice = slice(last_slice.start, slice_.stop, slice_.step)
            result[-1] = last_slice
        else:
            result.append(slice_)
            last_slice = slice_
    return result