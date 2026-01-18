from ._internal import NDArrayBase
from ..base import _Null
def _backward_gather_nd(data=None, indices=None, shape=_Null, out=None, name=None, **kwargs):
    """Accumulates data according to indices and get the result. It's the backward of
    `gather_nd`.

    Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape
    `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
    where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.

    The elements in output is defined as follows::

      output[indices[0, y_0, ..., y_{K-1}],
             ...,
             indices[M-1, y_0, ..., y_{K-1}],
             x_M, ..., x_{N-1}] += data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]

    all other entries in output are 0 or the original value if AddTo is triggered.

    Examples::

      data = [2, 3, 0]
      indices = [[1, 1, 0], [0, 1, 0]]
      shape = (2, 2)
      _backward_gather_nd(data, indices, shape) = [[0, 0], [2, 3]] # Same as scatter_nd

      # The difference between scatter_nd and scatter_nd_acc is the latter will accumulate
      #  the values that point to the same index.

      data = [2, 3, 0]
      indices = [[1, 1, 0], [1, 1, 0]]
      shape = (2, 2)
      _backward_gather_nd(data, indices, shape) = [[0, 0], [0, 5]]



    Parameters
    ----------
    data : NDArray
        data
    indices : NDArray
        indices
    shape : Shape(tuple), required
        Shape of output.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)