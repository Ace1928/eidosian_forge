import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def _broadcast_arrays(arrays, axis=None):
    """
    Broadcast shapes of arrays, ignoring incompatibility of specified axes
    """
    new_shapes = _broadcast_array_shapes(arrays, axis=axis)
    if axis is None:
        new_shapes = [new_shapes] * len(arrays)
    return [np.broadcast_to(array, new_shape) for array, new_shape in zip(arrays, new_shapes)]