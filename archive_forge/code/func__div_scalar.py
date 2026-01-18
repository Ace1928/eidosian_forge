from ._internal import NDArrayBase
from ..base import _Null
def _div_scalar(data=None, scalar=_Null, is_int=_Null, out=None, name=None, **kwargs):
    """Divide an array with a scalar.

    ``_div_scalar`` only operates on data array of input if input is sparse.

    For example, if input of shape (100, 100) has only 2 non zero elements,
    i.e. input.data = [5, 6], scalar = nan,
    it will result output.data = [nan, nan] instead of 10000 nans.



    Defined in ../src/operator/tensor/elemwise_binary_scalar_op_basic.cc:L174

    Parameters
    ----------
    data : NDArray
        source input
    scalar : double, optional, default=1
        Scalar input value
    is_int : boolean, optional, default=1
        Indicate whether scalar input is int type

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)