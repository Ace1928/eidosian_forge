from __future__ import annotations
import numpy as np
import xarray as xr
from .linalg import einsum, einsum_path, matmul
from .accessors import LinAlgAccessor, EinopsAccessor
def _remove_indexes_to_reduce(da, dims):
    """Remove indexes related to provided dims.

    Removes indexes related to dims on which we need to operate.
    As many functions only support integer `axis` or None,
    in order to have our functions operate on multiple dimensions
    we need to stack/flatten them. If some of those dimensions
    are already indexed by a multiindex this doesn't work, so we
    remove the indexes. As they are reduced, that information
    will end up being lost eventually either way.
    """
    index_keys = list(da.indexes)
    remove_indicator = [any((da.indexes[k] is index for k in index_keys if k in dims)) for name, index in da.indexes.items()]
    indexes_to_remove = [k for k, remove in zip(index_keys, remove_indicator) if remove]
    da = da.drop_indexes(indexes_to_remove)
    coords_to_remove = [coord for coord in da.coords if coord in indexes_to_remove or coord in dims]
    return da.reset_coords(coords_to_remove, drop=True)