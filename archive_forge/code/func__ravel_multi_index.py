from ._internal import NDArrayBase
from ..base import _Null
def _ravel_multi_index(data=None, shape=_Null, out=None, name=None, **kwargs):
    """Converts a batch of index arrays into an array of flat indices. The operator follows numpy conventions so a single multi index is given by a column of the input matrix. The leading dimension may be left unspecified by using -1 as placeholder.  

    Examples::
   
       A = [[3,6,6],[4,5,1]]
       ravel(A, shape=(7,6)) = [22,41,37]
       ravel(A, shape=(-1,6)) = [22,41,37]



    Defined in ../src/operator/tensor/ravel.cc:L41

    Parameters
    ----------
    data : NDArray
        Batch of multi-indices
    shape : Shape(tuple), optional, default=None
        Shape of the array into which the multi-indices apply.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)