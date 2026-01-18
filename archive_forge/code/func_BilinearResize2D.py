from ._internal import NDArrayBase
from ..base import _Null
def BilinearResize2D(data=None, like=None, height=_Null, width=_Null, scale_height=_Null, scale_width=_Null, mode=_Null, align_corners=_Null, out=None, name=None, **kwargs):
    """
    Perform 2D resizing (upsampling or downsampling) for 4D input using bilinear interpolation.

    Expected input is a 4 dimensional NDArray (NCHW) and the output
    with the shape of (N x C x height x width). 
    The key idea of bilinear interpolation is to perform linear interpolation
    first in one direction, and then again in the other direction. See the wikipedia of
    `Bilinear interpolation  <https://en.wikipedia.org/wiki/Bilinear_interpolation>`_
    for more details.


    Defined in ../src/operator/contrib/bilinear_resize.cc:L219

    Parameters
    ----------
    data : NDArray
        Input data
    like : NDArray
        Resize data to it's shape
    height : int, optional, default='1'
        output height (required, but ignored if scale_height is defined or mode is not "size")
    width : int, optional, default='1'
        output width (required, but ignored if scale_width is defined or mode is not "size")
    scale_height : float or None, optional, default=None
        sampling scale of the height (optional, used in modes "scale" and "odd_scale")
    scale_width : float or None, optional, default=None
        sampling scale of the width (optional, used in modes "scale" and "odd_scale")
    mode : {'like', 'odd_scale', 'size', 'to_even_down', 'to_even_up', 'to_odd_down', 'to_odd_up'},optional, default='size'
        resizing mode. "simple" - output height equals parameter "height" if "scale_height" parameter is not defined or input height multiplied by "scale_height" otherwise. Same for width;"odd_scale" - if original height or width is odd, then result height is calculated like result_h = (original_h - 1) * scale + 1; for scale > 1 the result shape would be like if we did deconvolution with kernel = (1, 1) and stride = (height_scale, width_scale); and for scale < 1 shape would be like we did convolution with kernel = (1, 1) and stride = (int(1 / height_scale), int( 1/ width_scale);"like" - resize first input to the height and width of second input; "to_even_down" - resize input to nearest lower even height and width (if original height is odd then result height = original height - 1);"to_even_up" - resize input to nearest bigger even height and width (if original height is odd then result height = original height + 1);"to_odd_down" - resize input to nearest odd height and width (if original height is odd then result height = original height - 1);"to_odd_up" - resize input to nearest odd height and width (if original height is odd then result height = original height + 1);
    align_corners : boolean, optional, default=1
        With align_corners = True, the interpolating doesn't proportionally align theoutput and input pixels, and thus the output values can depend on the input size.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)