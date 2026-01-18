from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = math_ops.cumsum(array_ops.ones(shape=shape, dtype=dtypes.int32), axis=-2)
    col_index = math_ops.cumsum(array_ops.ones(shape=shape, dtype=dtypes.int32), axis=-1)
    return math_ops.greater_equal(row_index, col_index)