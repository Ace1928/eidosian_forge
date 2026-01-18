from ._internal import NDArrayBase
from ..base import _Null
def _Native(*data, **kwargs):
    """Stub for implementing an operator implemented in native frontend language.

    Parameters
    ----------
    data : NDArray[]
        Input data for the custom operator.
    info : ptr, required
    need_top_grad : boolean, optional, default=1
        Whether this layer needs out grad for backward. Should be false for loss layers.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)