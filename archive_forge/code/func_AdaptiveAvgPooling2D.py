from ._internal import NDArrayBase
from ..base import _Null
def AdaptiveAvgPooling2D(data=None, output_size=_Null, out=None, name=None, **kwargs):
    """
    Applies a 2D adaptive average pooling over a 4D input with the shape of (NCHW).
    The pooling kernel and stride sizes are automatically chosen for desired output sizes.

    - If a single integer is provided for output_size, the output size is \\
      (N x C x output_size x output_size) for any input (NCHW).

    - If a tuple of integers (height, width) are provided for output_size, the output size is \\
      (N x C x height x width) for any input (NCHW).



    Defined in ../src/operator/contrib/adaptive_avg_pooling.cc:L213

    Parameters
    ----------
    data : NDArray
        Input data
    output_size : Shape(tuple), optional, default=[]
        int (output size) or a tuple of int for output (height, width).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)