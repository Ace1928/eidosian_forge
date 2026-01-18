from ._internal import NDArrayBase
from ..base import _Null
def _scatter_plus_scalar(data=None, scalar=_Null, is_int=_Null, out=None, name=None, **kwargs):
    """Adds a scalar to a tensor element-wise.  If the left-hand-side input is
    'row_sparse' or 'csr', then only the values which exist in the left-hand sparse array are computed.
    The 'missing' values are ignored.

    The storage type of ``_scatter_plus_scalar`` output depends on storage types of inputs

    - _scatter_plus_scalar(row_sparse, scalar) = row_sparse
    - _scatter_plus_scalar(csr, scalar) = csr
    - otherwise, ``_scatter_plus_scalar`` behaves exactly like _plus_scalar and generates output
    with default storage



    Parameters
    ----------
    data : NDArray
        source input
    scalar : double, optional, default=1
        Scalar input value
    is_int : boolean, optional, default=1
        Indicate whether scalar input is int type

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)