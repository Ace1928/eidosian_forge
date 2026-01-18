from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_config
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
def _broadcast_to_uniform_shape(rt_input, shape, broadcast_inner_dimensions):
    """Broadcasts rt_input to the uniform shape `shape`."""
    if isinstance(rt_input, ragged_tensor.RaggedTensor):
        raise ValueError('Incompatible with shape: ragged rank mismatch')
    if broadcast_inner_dimensions:
        return array_ops.broadcast_to(rt_input, shape.inner_dim_sizes)
    else:
        return rt_input