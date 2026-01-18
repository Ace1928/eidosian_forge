from ._internal import NDArrayBase
from ..base import _Null
def MultiBoxTarget(anchor=None, label=None, cls_pred=None, overlap_threshold=_Null, ignore_label=_Null, negative_mining_ratio=_Null, negative_mining_thresh=_Null, minimum_negative_samples=_Null, variances=_Null, out=None, name=None, **kwargs):
    """Compute Multibox training targets

    Parameters
    ----------
    anchor : NDArray
        Generated anchor boxes.
    label : NDArray
        Object detection labels.
    cls_pred : NDArray
        Class predictions.
    overlap_threshold : float, optional, default=0.5
        Anchor-GT overlap threshold to be regarded as a positive match.
    ignore_label : float, optional, default=-1
        Label for ignored anchors.
    negative_mining_ratio : float, optional, default=-1
        Max negative to positive samples ratio, use -1 to disable mining
    negative_mining_thresh : float, optional, default=0.5
        Threshold used for negative mining.
    minimum_negative_samples : int, optional, default='0'
        Minimum number of negative samples.
    variances : tuple of <float>, optional, default=[0.1,0.1,0.2,0.2]
        Variances to be encoded in box regression target.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)