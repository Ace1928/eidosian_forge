from __future__ import annotations
from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING, Generic, Literal, cast
import numpy as np
from numpy.typing import ArrayLike
from xarray.core import duck_array_ops, utils
from xarray.core.alignment import align, broadcast
from xarray.core.computation import apply_ufunc, dot
from xarray.core.types import Dims, T_DataArray, T_Xarray
from xarray.namedarray.utils import is_duck_dask_array
from xarray.util.deprecation_helpers import _deprecate_positional_args
def _check_dim(self, dim: Dims):
    """raise an error if any dimension is missing"""
    dims: list[Hashable]
    if isinstance(dim, str) or not isinstance(dim, Iterable):
        dims = [dim] if dim else []
    else:
        dims = list(dim)
    all_dims = set(self.obj.dims).union(set(self.weights.dims))
    missing_dims = set(dims) - all_dims
    if missing_dims:
        raise ValueError(f'Dimensions {tuple(missing_dims)} not found in {self.__class__.__name__} dimensions {tuple(all_dims)}')