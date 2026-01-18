from __future__ import annotations
import functools
import itertools
import math
import warnings
from collections.abc import Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
import numpy as np
from packaging.version import Version
from xarray.core import dtypes, duck_array_ops, utils
from xarray.core.arithmetic import CoarsenArithmetic
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.core.types import CoarsenBoundaryOptions, SideOptions, T_Xarray
from xarray.core.utils import (
from xarray.namedarray import pycompat
def _numbagg_reduce(self, func, keep_attrs, **kwargs):
    axis = self.obj.get_axis_num(self.dim[0])
    padded = self.obj.variable
    if self.center[0]:
        if is_duck_dask_array(padded.data):
            shift = -(self.window[0] + 1) // 2
            offset = (self.window[0] - 1) // 2
            valid = (slice(None),) * axis + (slice(offset, offset + self.obj.shape[axis]),)
        else:
            shift = -self.window[0] // 2 + 1
            valid = (slice(None),) * axis + (slice(-shift, None),)
        padded = padded.pad({self.dim[0]: (0, -shift)}, mode='constant')
    if is_duck_dask_array(padded.data) and False:
        raise AssertionError('should not be reachable')
    else:
        values = func(padded.data, window=self.window[0], min_count=self.min_periods, axis=axis)
    if self.center[0]:
        values = values[valid]
    attrs = self.obj.attrs if keep_attrs else {}
    return self.obj.__class__(values, self.obj.coords, attrs=attrs, name=self.obj.name)