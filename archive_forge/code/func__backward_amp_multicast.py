from ._internal import NDArrayBase
from ..base import _Null
def _backward_amp_multicast(*grad, **kwargs):
    """

    Parameters
    ----------
    grad : NDArray[]
        Gradients
    num_outputs : int, required
        Number of input/output pairs to be casted to the widest type.
    cast_narrow : boolean, optional, default=0
        Whether to cast to the narrowest type

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)