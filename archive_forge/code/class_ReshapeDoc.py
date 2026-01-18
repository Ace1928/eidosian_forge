import re as _re
from .base import build_param_doc as _build_param_doc
class ReshapeDoc(NDArrayDoc):
    """
    Examples
    --------
    Reshapes the input array into a new shape.

    >>> x = mx.nd.array([1, 2, 3, 4])
    >>> y = mx.nd.reshape(x, shape=(2, 2))
    >>> x.shape
    (4L,)
    >>> y.shape
    (2L, 2L)
    >>> y.asnumpy()
    array([[ 1.,  2.],
           [ 3.,  4.]], dtype=float32)

    You can use ``0`` to copy a particular dimension from the input to the output shape
    and '-1' to infer the dimensions of the output.

    >>> x = mx.nd.ones((2, 3, 4))
    >>> x.shape
    (2L, 3L, 4L)
    >>> y = mx.nd.reshape(x, shape=(4, 0, -1))
    >>> y.shape
    (4L, 3L, 2L)
    """