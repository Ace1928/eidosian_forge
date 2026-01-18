from ._internal import NDArrayBase
from ..base import _Null
def GridGenerator(data=None, transform_type=_Null, target_shape=_Null, out=None, name=None, **kwargs):
    """Generates 2D sampling grid for bilinear sampling.

    Parameters
    ----------
    data : NDArray
        Input data to the function.
    transform_type : {'affine', 'warp'}, required
        The type of transformation. For `affine`, input data should be an affine matrix of size (batch, 6). For `warp`, input data should be an optical flow of size (batch, 2, h, w).
    target_shape : Shape(tuple), optional, default=[0,0]
        Specifies the output shape (H, W). This is required if transformation type is `affine`. If transformation type is `warp`, this parameter is ignored.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)