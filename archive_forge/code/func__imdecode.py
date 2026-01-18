from ._internal import NDArrayBase
from ..base import _Null
def _imdecode(mean=None, index=_Null, x0=_Null, y0=_Null, x1=_Null, y1=_Null, c=_Null, size=_Null, out=None, name=None, **kwargs):
    """Decode an image, clip to (x0, y0, x1, y1), subtract mean, and write to buffer

    Parameters
    ----------
    mean : NDArray
        image mean
    index : int
        buffer position for output
    x0 : int
        x0
    y0 : int
        y0
    x1 : int
        x1
    y1 : int
        y1
    c : int
        channel
    size : int
        length of str_img

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)