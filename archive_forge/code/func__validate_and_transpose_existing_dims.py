from __future__ import annotations
import json
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding, conventions
from xarray.backends.common import (
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.types import ZarrWriteModes
from xarray.core.utils import (
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import guess_chunkmanager
from xarray.namedarray.pycompat import integer_types
def _validate_and_transpose_existing_dims(var_name, new_var, existing_var, region, append_dim):
    if new_var.dims != existing_var.dims:
        if set(existing_var.dims) == set(new_var.dims):
            new_var = new_var.transpose(*existing_var.dims)
        else:
            raise ValueError(f'variable {var_name!r} already exists with different dimension names {existing_var.dims} != {new_var.dims}, but changing variable dimensions is not supported by to_zarr().')
    existing_sizes = {}
    for dim, size in existing_var.sizes.items():
        if region is not None and dim in region:
            start, stop, stride = region[dim].indices(size)
            assert stride == 1
            size = stop - start
        if dim != append_dim:
            existing_sizes[dim] = size
    new_sizes = {dim: size for dim, size in new_var.sizes.items() if dim != append_dim}
    if existing_sizes != new_sizes:
        raise ValueError(f'variable {var_name!r} already exists with different dimension sizes: {existing_sizes} != {new_sizes}. to_zarr() only supports changing dimension sizes when explicitly appending, but append_dim={append_dim!r}. If you are attempting to write to a subset of the existing store without changing dimension sizes, consider using the region argument in to_zarr().')
    return new_var