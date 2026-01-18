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
def _broadcast_inner_dimension_to_uniform(self, axis, length):
    """Broadcasts the inner dimension `axis` to match `lengths`."""
    dim_size = self.dimension_size(axis)
    axis_in_inner_dims = axis - self.num_partitioned_dimensions
    partitioned_sizes = self._partitioned_dim_sizes
    inner_sizes = array_ops.concat([self._inner_dim_sizes[:axis_in_inner_dims], [array_ops.where(math_ops.equal(dim_size, 1), length, dim_size)], self._inner_dim_sizes[axis_in_inner_dims + 1:]], axis=0)
    return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes, self.dim_size_dtype)