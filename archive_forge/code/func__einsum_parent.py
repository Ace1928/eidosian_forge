import warnings
import numpy as np
import xarray as xr
def _einsum_parent(dims, *operands, keep_dims=frozenset()):
    """Preprocess inputs to call :func:`numpy.einsum` or :func:`numpy.einsum_path`.

    Parameters
    ----------
    dims : list of list of str
        List of lists of dimension names. It must have the same length or be
        only one item longer than ``operands``. If both have the same
        length, the generated pattern passed to {func}`numpy.einsum`
        won't have ``->`` nor right hand side. Otherwise, the last
        item is assumed to be the dimension specification of the output
        DataArray, and it can be an empty list to add ``->`` but no
        subscripts.
    operands : DataArray
        DataArrays for the operation. Multiple DataArrays are accepted.
    keep_dims : set, optional
        Dimensions to exclude from summation unless specifically specified in ``dims``

    See Also
    --------
    xarray_einstats.einsum, xarray_einstats.einsum_path
    numpy.einsum, numpy.einsum_path
    xarray_einstats.einops.reduce
    """
    if len(dims) == len(operands):
        in_dims = dims
        out_dims = None
    elif len(dims) == len(operands) + 1:
        in_dims = dims[:-1]
        out_dims = dims[-1]
    else:
        raise ValueError('length of dims and operands not compatible')
    all_dims = set((dim for sublist in dims for dim in sublist))
    handler = PairHandler(all_dims, keep_dims)
    in_subscripts = []
    updated_in_dims = []
    for da, sublist in zip(operands, in_dims):
        in_subs, up_dims = handler.process_dim_da_pair(da, sublist)
        in_subscripts.append(in_subs)
        updated_in_dims.append(up_dims)
    in_subscript = ','.join(in_subscripts)
    if out_dims is None:
        out_subscript = handler.get_out_subscript()
        out_dims = handler.out_dims
    elif not out_dims:
        out_subscript = '->'
    else:
        out_subscript = '->' + ''.join((handler.dim_map[dim] for dim in out_dims))
    if out_subscript and '...' in in_subscript:
        out_subscript = '->...' + out_subscript[2:]
    subscripts = in_subscript + out_subscript
    return (subscripts, updated_in_dims, out_dims)