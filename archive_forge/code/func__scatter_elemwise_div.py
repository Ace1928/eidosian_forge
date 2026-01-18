from ._internal import NDArrayBase
from ..base import _Null
def _scatter_elemwise_div(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Divides arguments element-wise.  If the left-hand-side input is 'row_sparse', then
    only the values which exist in the left-hand sparse array are computed.  The 'missing' values
    are ignored.

    The storage type of ``_scatter_elemwise_div`` output depends on storage types of inputs

    - _scatter_elemwise_div(row_sparse, row_sparse) = row_sparse
    - _scatter_elemwise_div(row_sparse, dense) = row_sparse
    - _scatter_elemwise_div(row_sparse, csr) = row_sparse
    - otherwise, ``_scatter_elemwise_div`` behaves exactly like elemwise_div and generates output
    with default storage



    Parameters
    ----------
    lhs : NDArray
        first input
    rhs : NDArray
        second input

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)