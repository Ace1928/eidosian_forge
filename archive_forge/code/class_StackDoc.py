import re as _re
from .base import build_param_doc as _build_param_doc
class StackDoc(NDArrayDoc):
    """
    Example
    --------
    Join a sequence of arrays along a new axis.
    >>> x = mx.nd.array([1, 2])
    >>> y = mx.nd.array([3, 4])
    >>> stack(x, y)
    [[1, 2],
     [3, 4]]
    >>> stack(x, y, axis=1)
    [[1, 3],
     [2, 4]]
    """