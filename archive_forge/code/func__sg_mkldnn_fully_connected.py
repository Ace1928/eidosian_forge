from ._internal import NDArrayBase
from ..base import _Null
def _sg_mkldnn_fully_connected(out=None, name=None, **kwargs):
    """_sg_mkldnn_fully_connected

    Defined in ../src/operator/subgraph/mkldnn/mkldnn_fc.cc:L636

    Parameters
    ----------


    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)