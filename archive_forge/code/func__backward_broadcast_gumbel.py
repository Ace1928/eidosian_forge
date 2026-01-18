from ._internal import NDArrayBase
from ..base import _Null
def _backward_broadcast_gumbel(loc=_Null, scale=_Null, size=_Null, ctx=_Null, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    loc : float or None, required
    scale : float or None, required
    size : Shape or None, optional, default=None
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
    ctx : string, optional, default='cpu'
        Context of output, in format [cpu|gpu|cpu_pinned](n). Only used for imperative calls.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)