from ._internal import NDArrayBase
from ..base import _Null
def ROIAlign(data=None, rois=None, pooled_size=_Null, spatial_scale=_Null, sample_ratio=_Null, position_sensitive=_Null, aligned=_Null, out=None, name=None, **kwargs):
    """
    This operator takes a 4D feature map as an input array and region proposals as `rois`,
    then align the feature map over sub-regions of input and produces a fixed-sized output array.
    This operator is typically used in Faster R-CNN & Mask R-CNN networks. If roi batchid is less 
    than 0, it will be ignored, and the corresponding output will be set to 0.

    Different from ROI pooling, ROI Align removes the harsh quantization, properly aligning
    the extracted features with the input. RoIAlign computes the value of each sampling point
    by bilinear interpolation from the nearby grid points on the feature map. No quantization is
    performed on any coordinates involved in the RoI, its bins, or the sampling points.
    Bilinear interpolation is used to compute the exact values of the
    input features at four regularly sampled locations in each RoI bin.
    Then the feature map can be aggregated by avgpooling.


    References
    ----------

    He, Kaiming, et al. "Mask R-CNN." ICCV, 2017


    Defined in ../src/operator/contrib/roi_align.cc:L558

    Parameters
    ----------
    data : NDArray
        Input data to the pooling operator, a 4D Feature maps
    rois : NDArray
        Bounding box coordinates, a 2D array, if batchid is less than 0, it will be ignored.
    pooled_size : Shape(tuple), required
        ROI Align output roi feature map height and width: (h, w)
    spatial_scale : float, required
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers
    sample_ratio : int, optional, default='-1'
        Optional sampling ratio of ROI align, using adaptive size by default.
    position_sensitive : boolean, optional, default=0
        Whether to perform position-sensitive RoI pooling. PSRoIPooling is first proposaled by R-FCN and it can reduce the input channels by ph*pw times, where (ph, pw) is the pooled_size
    aligned : boolean, optional, default=0
        Center-aligned ROIAlign introduced in Detectron2. To enable, set aligned to True.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)