from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_ragged_conversion_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
@ops.RegisterGradient('RaggedTensorFromVariant')
def _ragged_tensor_from_variant_grad(op, *grads):
    """Gradient for RaggedTensorFromVariant op."""
    variant_rank = op.inputs[0].shape.rank
    if variant_rank == 0:
        batched_input = False
    elif variant_rank == 1:
        batched_input = True
    elif variant_rank is None:
        batched_input = op.get_attr('output_ragged_rank') > 0
    else:
        raise ValueError('Unable to compute gradient: RaggedTensorToVariant can currently only generate 0D or 1D output.')
    return [gen_ragged_conversion_ops.ragged_tensor_to_variant(rt_nested_splits=op.outputs[:-1], rt_dense_values=grads[-1], batched_input=batched_input)]