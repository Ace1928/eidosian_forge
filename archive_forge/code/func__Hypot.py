from ._internal import NDArrayBase
from ..base import _Null
def _Hypot(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """Given the "legs" of a right triangle, return its hypotenuse.



    Defined in ../src/operator/tensor/elemwise_binary_op_extended.cc:L78

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