from ._internal import NDArrayBase
from ..base import _Null
def _cvimdecode(buf=None, flag=_Null, to_rgb=_Null, out=None, name=None, **kwargs):
    """Decode image with OpenCV. 
    Note: return image in RGB by default, instead of OpenCV's default BGR.

    Parameters
    ----------
    buf : NDArray
        Buffer containing binary encoded image
    flag : int, optional, default='1'
        Convert decoded image to grayscale (0) or color (1).
    to_rgb : boolean, optional, default=1
        Whether to convert decoded image to mxnet's default RGB format (instead of opencv's default BGR).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)