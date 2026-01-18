from ._internal import NDArrayBase
from ..base import _Null
def ElementWiseSum(*args, **kwargs):
    """Adds all input arguments element-wise.

    .. math::
       add\\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n

    ``add_n`` is potentially more efficient than calling ``add`` by `n` times.

    The storage type of ``add_n`` output depends on storage types of inputs

    - add_n(row_sparse, row_sparse, ..) = row_sparse
    - add_n(default, csr, default) = default
    - add_n(any input combinations longer than 4 (>4) with at least one default type) = default
    - otherwise, ``add_n`` falls all inputs back to default storage and generates default storage



    Defined in ../src/operator/tensor/elemwise_sum.cc:L155

    Parameters
    ----------
    args : NDArray[]
        Positional input arguments

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)