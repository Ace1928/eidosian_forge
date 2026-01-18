from __future__ import annotations
import copy
import itertools
import math
import numbers
import warnings
from collections.abc import Hashable, Mapping, Sequence
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, NoReturn, cast
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
import xarray as xr  # only for Dataset and DataArray
from xarray.core import common, dtypes, duck_array_ops, indexing, nputils, ops, utils
from xarray.core.arithmetic import VariableArithmetic
from xarray.core.common import AbstractArray
from xarray.core.indexing import (
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.utils import (
from xarray.namedarray.core import NamedArray, _raise_if_any_duplicate_dimensions
from xarray.namedarray.pycompat import integer_types, is_0d_dask_array, to_duck_array
def _broadcast_compat_data(self, other):
    if not OPTIONS['arithmetic_broadcast']:
        if isinstance(other, Variable) and self.dims != other.dims or (is_duck_array(other) and self.ndim != other.ndim):
            raise ValueError("Broadcasting is necessary but automatic broadcasting is disabled via global option `'arithmetic_broadcast'`. Use `xr.set_options(arithmetic_broadcast=True)` to enable automatic broadcasting.")
    if all((hasattr(other, attr) for attr in ['dims', 'data', 'shape', 'encoding'])):
        new_self, new_other = _broadcast_compat_variables(self, other)
        self_data = new_self.data
        other_data = new_other.data
        dims = new_self.dims
    else:
        self_data = self.data
        other_data = other
        dims = self.dims
    return (self_data, other_data, dims)