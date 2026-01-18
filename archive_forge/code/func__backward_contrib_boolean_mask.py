from ._internal import NDArrayBase
from ..base import _Null
def _backward_contrib_boolean_mask(axis=_Null, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    axis : int, optional, default='0'
        An integer that represents the axis in NDArray to mask from.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)