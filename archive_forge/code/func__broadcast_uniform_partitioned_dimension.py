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
def _broadcast_uniform_partitioned_dimension(self, axis, lengths):
    """Broadcasts the partitioned dimension `axis` to match `lengths`."""
    axis_dim_size = self.dimension_size(axis)
    partitioned_sizes = list(self._partitioned_dim_sizes[:axis])
    if lengths.shape.ndims == 0:
        lengths = array_ops.where(math_ops.equal(axis_dim_size, 1), lengths, axis_dim_size)
        repeats = array_ops.where(math_ops.equal(axis_dim_size, 1), lengths, 1)
        splits = array_ops_stack.stack([0, self.num_slices_in_dimension(axis)])
    else:
        splits = math_ops.range(array_ops.size(lengths, out_type=self.dim_size_dtype) + 1)
        repeats = lengths
    partitioned_sizes.append(lengths)
    for dim_size in self._partitioned_dim_sizes[axis + 1:]:
        if dim_size.shape.ndims == 0:
            partitioned_sizes.append(dim_size)
            splits *= dim_size
        else:
            partitioned_sizes.append(ragged_util.repeat_ranges(dim_size, splits, repeats))
            splits = array_ops.gather(ragged_util.lengths_to_splits(dim_size), splits)
    inner_sizes = self._inner_dim_sizes
    return RaggedTensorDynamicShape(partitioned_sizes, inner_sizes, self.dim_size_dtype)