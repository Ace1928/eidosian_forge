from ._internal import NDArrayBase
from ..base import _Null
def RROIAlign(data=None, rois=None, pooled_size=_Null, spatial_scale=_Null, sampling_ratio=_Null, out=None, name=None, **kwargs):
    """Performs Rotated ROI Align on the input array.

    This operator takes a 4D feature map as an input array and region proposals as `rois`,
    then align the feature map over sub-regions of input and produces a fixed-sized output array.

    Different from ROI Align, RROI Align uses rotated rois, which is suitable for text detection.
    RRoIAlign computes the value of each sampling point by bilinear interpolation from the nearby
    grid points on the rotated feature map. No quantization is performed on any coordinates
    involved in the RoI, its bins, or the sampling points. Bilinear interpolation is used to
    compute the exact values of the input features at four regularly sampled locations in
    each RoI bin. Then the feature map can be aggregated by avgpooling.

    References
    ----------

    Ma, Jianqi, et al. "Arbitrary-Oriented Scene Text Detection via Rotation Proposals."
    IEEE Transactions on Multimedia, 2018.



    Defined in ../src/operator/contrib/rroi_align.cc:L273

    Parameters
    ----------
    data : NDArray
        Input data to the pooling operator, a 4D Feature maps
    rois : NDArray
        Bounding box coordinates, a 2D array
    pooled_size : Shape(tuple), required
        RROI align output shape (h,w) 
    spatial_scale : float, required
        Ratio of input feature map height (or width) to raw image height (or width). Equals the reciprocal of total stride in convolutional layers
    sampling_ratio : int, optional, default='-1'
        Optional sampling ratio of RROI align, using adaptive size by default.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)