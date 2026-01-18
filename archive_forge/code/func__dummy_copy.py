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
def _dummy_copy(xarray_obj):
    from xarray.core.dataarray import DataArray
    from xarray.core.dataset import Dataset
    if isinstance(xarray_obj, Dataset):
        res = Dataset({k: dtypes.get_fill_value(v.dtype) for k, v in xarray_obj.data_vars.items()}, {k: dtypes.get_fill_value(v.dtype) for k, v in xarray_obj.coords.items() if k not in xarray_obj.dims}, xarray_obj.attrs)
    elif isinstance(xarray_obj, DataArray):
        res = DataArray(dtypes.get_fill_value(xarray_obj.dtype), {k: dtypes.get_fill_value(v.dtype) for k, v in xarray_obj.coords.items() if k not in xarray_obj.dims}, dims=[], name=xarray_obj.name, attrs=xarray_obj.attrs)
    else:
        raise AssertionError
    return res