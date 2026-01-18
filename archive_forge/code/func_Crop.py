from ._internal import NDArrayBase
from ..base import _Null
def Crop(*data, **kwargs):
    """

    .. note:: `Crop` is deprecated. Use `slice` instead.

    Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
    with width and height of the second input symbol, i.e., with one input, we need h_w to
    specify the crop height and width, otherwise the second input symbol's size will be used


    Defined in ../src/operator/crop.cc:L49

    Parameters
    ----------
    data : Symbol or Symbol[]
        Tensor or List of Tensors, the second input will be used as crop_like shape reference
    offset : Shape(tuple), optional, default=[0,0]
        crop offset coordinate: (y, x)
    h_w : Shape(tuple), optional, default=[0,0]
        crop height and width: (h, w)
    center_crop : boolean, optional, default=0
        If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)