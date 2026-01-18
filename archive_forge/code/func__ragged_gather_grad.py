from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch
@ops.RegisterGradient('RaggedGather')
def _ragged_gather_grad(op, *grads):
    """Gradient for RaggedGather op."""
    param_nested_splits = op.inputs[:-2]
    param_inner_values = op.inputs[-2]
    indices = op.inputs[-1]
    grad_inner_values = grads[-1]
    combined_splits = param_nested_splits[0]
    for row_splits in param_nested_splits[1:]:
        combined_splits = array_ops.gather(row_splits, combined_splits)
    flat_indices = array_ops.reshape(indices, [-1])
    grad_indices = ragged_math_ops.range(array_ops.gather(combined_splits, flat_indices), array_ops.gather(combined_splits[1:], flat_indices)).values
    param_inner_values_grad = indexed_slices.IndexedSlices(values=grad_inner_values, indices=grad_indices, dense_shape=array_ops.shape(param_inner_values))
    return [None for _ in param_nested_splits] + [param_inner_values_grad, None]