import six
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def _is_shape_and_default_value_compatible(default_value, shape):
    """Verifies compatibility of shape and default_value."""
    if nest.is_nested(default_value) != bool(shape):
        return False
    if not shape:
        return True
    if len(default_value) != shape[0]:
        return False
    for i in range(shape[0]):
        if not _is_shape_and_default_value_compatible(default_value[i], shape[1:]):
            return False
    return True