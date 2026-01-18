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
def _shift_one_dim(self, dim, count, fill_value=dtypes.NA):
    axis = self.get_axis_num(dim)
    if count > 0:
        keep = slice(None, -count)
    elif count < 0:
        keep = slice(-count, None)
    else:
        keep = slice(None)
    trimmed_data = self[(slice(None),) * axis + (keep,)].data
    if fill_value is dtypes.NA:
        dtype, fill_value = dtypes.maybe_promote(self.dtype)
    else:
        dtype = self.dtype
    width = min(abs(count), self.shape[axis])
    dim_pad = (width, 0) if count >= 0 else (0, width)
    pads = [(0, 0) if d != dim else dim_pad for d in self.dims]
    data = np.pad(duck_array_ops.astype(trimmed_data, dtype), pads, mode='constant', constant_values=fill_value)
    if is_duck_dask_array(data):
        data = data.rechunk(self.data.chunks)
    return self._replace(data=data)