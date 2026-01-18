from ._internal import NDArrayBase
from ..base import _Null
def _crop_assign(lhs=None, rhs=None, begin=_Null, end=_Null, step=_Null, out=None, name=None, **kwargs):
    """Assign the rhs to a cropped subset of lhs.

    Requirements
    ------------
    - output should be explicitly given and be the same as lhs.
    - lhs and rhs are of the same data type, and on the same device.


    From:../src/operator/tensor/matrix_op.cc:514

    Parameters
    ----------
    lhs : NDArray
        Source input
    rhs : NDArray
        value to assign
    begin : Shape(tuple), required
        starting indices for the slice operation, supports negative indices.
    end : Shape(tuple), required
        ending indices for the slice operation, supports negative indices.
    step : Shape(tuple), optional, default=[]
        step for the slice operation, supports negative values.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)