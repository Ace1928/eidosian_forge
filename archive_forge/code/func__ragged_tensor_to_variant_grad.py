from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
@ops.RegisterGradient('RaggedTensorToVariant')
def _ragged_tensor_to_variant_grad(op, encoded_ragged_grad):
    """Gradient for RaggedTensorToVariant op."""
    dense_values = op.inputs[-1]
    ragged_rank = len(op.inputs) - 1
    row_splits = 0 if ragged_rank == 0 else op.inputs[0]
    values_grad = gen_ragged_conversion_ops.ragged_tensor_to_variant_gradient(encoded_ragged_grad=encoded_ragged_grad, row_splits=row_splits, dense_values_shape=array_ops.shape(dense_values), Tvalues=op.inputs[-1].dtype)
    result = [None] * ragged_rank + [values_grad]
    return result