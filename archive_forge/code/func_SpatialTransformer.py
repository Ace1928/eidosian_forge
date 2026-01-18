from ._internal import NDArrayBase
from ..base import _Null
def SpatialTransformer(data=None, loc=None, target_shape=_Null, transform_type=_Null, sampler_type=_Null, cudnn_off=_Null, out=None, name=None, **kwargs):
    """Applies a spatial transformer to input feature map.

    Parameters
    ----------
    data : NDArray
        Input data to the SpatialTransformerOp.
    loc : NDArray
        localisation net, the output dim should be 6 when transform_type is affine. You shold initialize the weight and bias with identity tranform.
    target_shape : Shape(tuple), optional, default=[0,0]
        output shape(h, w) of spatial transformer: (y, x)
    transform_type : {'affine'}, required
        transformation type
    sampler_type : {'bilinear'}, required
        sampling type
    cudnn_off : boolean or None, optional, default=None
        whether to turn cudnn off

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)