from ._internal import NDArrayBase
from ..base import _Null
def _NDArray(*data, **kwargs):
    """Stub for implementing an operator implemented in native frontend language with ndarray.

    Parameters
    ----------
    data : NDArray[]
        Input data for the custom operator.
    info : ptr, required

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)