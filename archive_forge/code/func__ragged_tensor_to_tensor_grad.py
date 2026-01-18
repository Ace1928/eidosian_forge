from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
@ops.RegisterGradient('RaggedTensorToTensor')
def _ragged_tensor_to_tensor_grad(op, grad):
    """Gradient for RaggedToTensor op."""
    flat_values = op.inputs[1]
    default_value = op.inputs[2]
    row_partition_tensors = op.inputs[3:]
    row_partition_types = op.get_attr('row_partition_types')
    flat_value_shape = array_ops.shape(flat_values)
    ragged_rank = sum((1 for typ in row_partition_types if typ != b'FIRST_DIM_SIZE'))
    indices = gen_ragged_conversion_ops.ragged_tensor_to_tensor(shape=array_ops.shape(grad)[:1 + ragged_rank], values=math_ops.range(flat_value_shape[0]), default_value=-1, row_partition_types=row_partition_types, row_partition_tensors=row_partition_tensors)
    mask = math_ops.not_equal(indices, -1)
    values_grad = indexed_slices.IndexedSlices(values=array_ops.boolean_mask(grad, mask), indices=array_ops.boolean_mask(indices, mask), dense_shape=flat_value_shape)
    default_grads = array_ops.boolean_mask(grad, ~mask)
    dims_to_reduce = math_ops.range(array_ops.rank(default_grads) - _rank_ignoring_leading_dims_with_size_1(default_value))
    default_grad = math_ops.reduce_sum(default_grads, axis=dims_to_reduce)
    default_grad = array_ops.reshape(default_grad, array_ops.shape(default_value))
    return [None, values_grad, default_grad] + [None for _ in row_partition_tensors]