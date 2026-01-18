from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices as indexed_slices_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
@ops.RegisterGradient('Tile')
def _TileGrad(op, grad):
    """Sum reduces grad along the tiled dimensions."""
    input_shape = array_ops.shape(op.inputs[0], out_type=op.inputs[1].dtype)
    split_shape = array_ops.reshape(array_ops.transpose(array_ops_stack.stack([op.inputs[1], input_shape])), [-1])
    axes = math_ops.range(0, array_ops.size(split_shape), 2)
    if isinstance(grad, indexed_slices_lib.IndexedSlices):
        input_shape_0 = math_ops.cast(input_shape[0], grad.indices.dtype)
        grad = math_ops.unsorted_segment_sum(grad.values, math_ops.mod(grad.indices, input_shape_0), input_shape_0)
        split_shape = array_ops.concat([[1], split_shape[1:]], axis=0)
    input_grad = math_ops.reduce_sum(array_ops.reshape(grad, split_shape), axes)
    if not context.executing_eagerly():
        input_grad.set_shape(op.inputs[0].get_shape())
    return [input_grad, None]