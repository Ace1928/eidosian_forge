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
def _array_reduce(self, array_agg_func, bottleneck_move_func, rolling_agg_func, keep_attrs, **kwargs):
    return self._dataset_implementation(functools.partial(DataArrayRolling._array_reduce, array_agg_func=array_agg_func, bottleneck_move_func=bottleneck_move_func, rolling_agg_func=rolling_agg_func), keep_attrs=keep_attrs, **kwargs)