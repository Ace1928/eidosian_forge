from ._internal import NDArrayBase
from ..base import _Null
def _identity_with_attr_like_rhs(lhs=None, rhs=None, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    lhs : NDArray
        First input.
    rhs : NDArray
        Second input.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)