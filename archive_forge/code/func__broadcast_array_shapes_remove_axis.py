import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def _broadcast_array_shapes_remove_axis(arrays, axis=None):
    """
    Broadcast shapes of arrays, dropping specified axes

    Given a sequence of arrays `arrays` and an integer or tuple `axis`, find
    the shape of the broadcast result after consuming/dropping `axis`.
    In other words, return output shape of a typical hypothesis test on
    `arrays` vectorized along `axis`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats._axis_nan_policy import _broadcast_array_shapes
    >>> a = np.zeros((5, 2, 1))
    >>> b = np.zeros((9, 3))
    >>> _broadcast_array_shapes((a, b), 1)
    (5, 3)
    """
    shapes = [arr.shape for arr in arrays]
    return _broadcast_shapes_remove_axis(shapes, axis)