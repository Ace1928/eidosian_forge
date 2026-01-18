from ._internal import NDArrayBase
from ..base import _Null
def _backward_contrib_bipartite_matching(is_ascend=_Null, threshold=_Null, topk=_Null, out=None, name=None, **kwargs):
    """

    Parameters
    ----------
    is_ascend : boolean, optional, default=0
        Use ascend order for scores instead of descending. Please set threshold accordingly.
    threshold : float, required
        Ignore matching when score < thresh, if is_ascend=false, or ignore score > thresh, if is_ascend=true.
    topk : int, optional, default='-1'
        Limit the number of matches to topk, set -1 for no limit

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)