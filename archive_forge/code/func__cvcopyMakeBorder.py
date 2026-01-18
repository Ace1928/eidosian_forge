from ._internal import NDArrayBase
from ..base import _Null
def _cvcopyMakeBorder(src=None, top=_Null, bot=_Null, left=_Null, right=_Null, type=_Null, value=_Null, values=_Null, out=None, name=None, **kwargs):
    """Pad image border with OpenCV. 


    Parameters
    ----------
    src : NDArray
        source image
    top : int, required
        Top margin.
    bot : int, required
        Bottom margin.
    left : int, required
        Left margin.
    right : int, required
        Right margin.
    type : int, optional, default='0'
        Filling type (default=cv2.BORDER_CONSTANT).
    value : double, optional, default=0
        (Deprecated! Use ``values`` instead.) Fill with single value.
    values : tuple of <double>, optional, default=[]
        Fill with value(RGB[A] or gray), up to 4 channels.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)