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
@ops.RegisterGradient('BroadcastTo')
def _BroadcastToGrad(op, grad):
    input_value = op.inputs[0]
    broadcast_shape = op.inputs[1]
    shape_dtype = dtypes.int32
    if isinstance(broadcast_shape, tensor.Tensor):
        shape_dtype = broadcast_shape.dtype
    input_value_shape = array_ops.shape(input_value, out_type=shape_dtype)
    if not isinstance(broadcast_shape, ops.EagerTensor):
        broadcast_shape_static = tensor_shape.TensorShape(tensor_util.try_evaluate_constant(broadcast_shape))
        if broadcast_shape_static.is_fully_defined():
            broadcast_shape = constant_op.constant(broadcast_shape_static.as_list(), dtype=shape_dtype)
    _, reduction_axes = gen_array_ops.broadcast_gradient_args(broadcast_shape, input_value_shape)
    updates_grad_reshaped = math_ops.reduce_sum(grad, axis=reduction_axes, keepdims=True)
    updates_grad = array_ops.reshape(updates_grad_reshaped, input_value_shape)
    return [updates_grad, None]