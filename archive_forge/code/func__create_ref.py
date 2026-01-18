from __future__ import annotations
import numpy as np
import xarray as xr
from .linalg import einsum, einsum_path, matmul
from .accessors import LinAlgAccessor, EinopsAccessor
def _create_ref(*args, dims, np_creator, dtype=None):
    if dtype is None and all((isinstance(arg, xr.DataArray) for arg in args)):
        dtype = np.result_type(*[arg.dtype for arg in args])
    ref_idxs = [_find_index(dim, args) for dim in dims]
    shape = [len(args[idx][dim]) for idx, dim in ref_idxs]
    coords = {dim: args[idx][dim] for idx, dim in ref_idxs if dim in args[idx].coords}
    return xr.DataArray(np_creator(shape, dtype=dtype), dims=dims, coords=coords)