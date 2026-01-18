from ._internal import NDArrayBase
from ..base import _Null
def _crop_assign_scalar(data=None, scalar=_Null, begin=_Null, end=_Null, step=_Null, out=None, name=None, **kwargs):
    """(Assign the scalar to a cropped subset of the input.

    Requirements
    ------------
    - output should be explicitly given and be the same as input
    )

    From:../src/operator/tensor/matrix_op.cc:540

    Parameters
    ----------
    data : NDArray
        Source input
    scalar : double, optional, default=0
        The scalar value for assignment.
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