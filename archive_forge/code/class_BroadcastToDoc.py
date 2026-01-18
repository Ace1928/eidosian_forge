import re as _re
from .base import build_param_doc as _build_param_doc
class BroadcastToDoc(NDArrayDoc):
    """
    Examples
    --------
    Broadcasts the input array into a new shape.
    >>> a = mx.nd.array(np.arange(6).reshape(6,1))
    >>> b = a.broadcast_to((6,2))
    >>> a.shape
    (6L, 1L)
    >>> b.shape
    (6L, 2L)
    >>> b.asnumpy()
    array([[ 0.,  0.],
       [ 1.,  1.],
       [ 2.,  2.],
       [ 3.,  3.],
       [ 4.,  4.],
       [ 5.,  5.]], dtype=float32)
    Broadcasts along axes 1 and 2.
    >>> c = a.reshape((2,1,1,3))
    >>> d = c.broadcast_to((2,2,2,3))
    >>> d.asnumpy()
    array([[[[ 0.,  1.,  2.],
         [ 0.,  1.,  2.]],

        [[ 0.,  1.,  2.],
         [ 0.,  1.,  2.]]],


       [[[ 3.,  4.,  5.],
         [ 3.,  4.,  5.]],

        [[ 3.,  4.,  5.],
         [ 3.,  4.,  5.]]]], dtype=float32)
    >>> c.shape
    (2L, 1L, 1L, 3L)
    >>> d.shape
    (2L, 2L, 2L, 3L)
    """