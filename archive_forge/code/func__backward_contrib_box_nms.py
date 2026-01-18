from ._internal import NDArrayBase
from ..base import _Null
def _backward_contrib_box_nms(overlap_thresh=_Null, valid_thresh=_Null, topk=_Null, coord_start=_Null, score_index=_Null, id_index=_Null, background_id=_Null, force_suppress=_Null, in_format=_Null, out_format=_Null, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    overlap_thresh : float, optional, default=0.5
        Overlapping(IoU) threshold to suppress object with smaller score.
    valid_thresh : float, optional, default=0
        Filter input boxes to those whose scores greater than valid_thresh.
    topk : int, optional, default='-1'
        Apply nms to topk boxes with descending scores, -1 to no restriction.
    coord_start : int, optional, default='2'
        Start index of the consecutive 4 coordinates.
    score_index : int, optional, default='1'
        Index of the scores/confidence of boxes.
    id_index : int, optional, default='-1'
        Optional, index of the class categories, -1 to disable.
    background_id : int, optional, default='-1'
        Optional, id of the background class which will be ignored in nms.
    force_suppress : boolean, optional, default=0
        Optional, if set false and id_index is provided, nms will only apply to boxes belongs to the same category
    in_format : {'center', 'corner'},optional, default='corner'
        The input box encoding type. 
     "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].
    out_format : {'center', 'corner'},optional, default='corner'
        The output box encoding type. 
     "corner" means boxes are encoded as [xmin, ymin, xmax, ymax], "center" means boxes are encodes as [x, y, width, height].

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)