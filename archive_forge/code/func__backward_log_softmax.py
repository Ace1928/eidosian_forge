from ._internal import NDArrayBase
from ..base import _Null
def _backward_log_softmax(*args, **kwargs):
    """

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