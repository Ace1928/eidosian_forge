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
def _broadcast_inner_dimension_to_ragged(self, axis, lengths):
    axis_in_inner_dims = axis - self.num_partitioned_dimensions
    partitioned_sizes = self._partitioned_dim_sizes + tuple([self._inner_dim_sizes[i] for i in range(axis_in_inner_dims)]) + (lengths,)
    inner_sizes = self._inner_dim_sizes[axis_in_inner_dims + 1:]
    return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes)