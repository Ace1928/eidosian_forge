import warnings
import numpy as np
import xarray as xr
def _einsum_path(dims, *operands, keep_dims=frozenset(), optimize=None, **kwargs):
    """Wrap :func:`numpy.einsum_path` directly."""
    op_kwargs = {} if optimize is None else {'optimize': optimize}
    subscripts, in_dims, _ = _einsum_parent(dims, *operands, keep_dims=keep_dims)
    updated_in_dims = []
    for sublist, da in zip(in_dims, operands):
        updated_in_dims.append([dim for dim in da.dims if dim not in sublist] + sublist)
    return xr.apply_ufunc(np.einsum_path, subscripts, *operands, input_core_dims=[[], *updated_in_dims], output_core_dims=[[]], kwargs=op_kwargs, **kwargs).values.item()