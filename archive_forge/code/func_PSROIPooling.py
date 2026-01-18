from ._internal import NDArrayBase
from ..base import _Null
def PSROIPooling(data=None, rois=None, spatial_scale=_Null, output_dim=_Null, pooled_size=_Null, group_size=_Null, out=None, name=None, **kwargs):
    """Performs region-of-interest pooling on inputs. Resize bounding box coordinates by spatial_scale and crop input feature maps accordingly. The cropped feature maps are pooled by max pooling to a fixed size output indicated by pooled_size. batch_size will change to the number of region bounding boxes after PSROIPooling

    Parameters
    ----------
    data : Symbol
        Input data to the pooling operator, a 4D Feature maps
    rois : Symbol
        Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]]. (x1, y1) and (x2, y2) are top left and down right corners of designated region of interest. batch_index indicates the index of corresponding image in the input data
    spatial_scale : float, required
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers
    output_dim : int, required
        fix output dim
    pooled_size : int, required
        fix pooled size
    group_size : int, optional, default='0'
        fix group size

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)