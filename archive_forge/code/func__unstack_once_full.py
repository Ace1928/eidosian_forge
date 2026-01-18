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
def _unstack_once_full(self, dim: Mapping[Any, int], old_dim: Hashable) -> Self:
    """
        Unstacks the variable without needing an index.

        Unlike `_unstack_once`, this function requires the existing dimension to
        contain the full product of the new dimensions.
        """
    new_dim_names = tuple(dim.keys())
    new_dim_sizes = tuple(dim.values())
    if old_dim not in self.dims:
        raise ValueError(f'invalid existing dimension: {old_dim}')
    if set(new_dim_names).intersection(self.dims):
        raise ValueError('cannot create a new dimension with the same name as an existing dimension')
    if math.prod(new_dim_sizes) != self.sizes[old_dim]:
        raise ValueError('the product of the new dimension sizes must equal the size of the old dimension')
    other_dims = [d for d in self.dims if d != old_dim]
    dim_order = other_dims + [old_dim]
    reordered = self.transpose(*dim_order)
    new_shape = reordered.shape[:len(other_dims)] + new_dim_sizes
    new_data = duck_array_ops.reshape(reordered.data, new_shape)
    new_dims = reordered.dims[:len(other_dims)] + new_dim_names
    return type(self)(new_dims, new_data, self._attrs, self._encoding, fastpath=True)