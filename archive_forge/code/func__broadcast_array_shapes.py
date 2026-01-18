import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def _broadcast_array_shapes(arrays, axis=None):
    """
    Broadcast shapes of arrays, ignoring incompatibility of specified axes
    """
    shapes = [np.asarray(arr).shape for arr in arrays]
    return _broadcast_shapes(shapes, axis)