from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
def _transpose_batch_time(x):
    """Transposes the batch and time dimensions of a Tensor.

  If the input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A Tensor.

  Returns:
    x transposed along the first two dimensions.
  """
    x_static_shape = x.get_shape()
    if x_static_shape.rank is not None and x_static_shape.rank < 2:
        return x
    x_rank = array_ops.rank(x)
    x_t = array_ops.transpose(x, array_ops.concat(([1, 0], math_ops.range(2, x_rank)), axis=0))
    x_t.set_shape(tensor_shape.TensorShape([x_static_shape.dims[1].value, x_static_shape.dims[0].value]).concatenate(x_static_shape[2:]))
    return x_t