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
def _TensorScatterMinOrMaxGrad(op, grad):
    """Gradient for TensorScatterMin and TensorScatterMax."""
    indices = op.inputs[1]
    x = op.inputs[0]
    y = op.inputs[2]
    output = op.outputs[0]
    x_indicators = math_ops.cast(math_ops.equal(x, output), grad.dtype)
    y_output = array_ops.gather_nd(output, indices)
    y_indicators = math_ops.cast(math_ops.equal(y, y_output), grad.dtype)
    ys_indicators = array_ops.scatter_nd(indices, y_indicators, array_ops.shape(x, out_type=indices.dtype))
    indicators = x_indicators + ys_indicators
    x_grad = grad * x_indicators / indicators
    y_grad = array_ops.gather_nd(grad / indicators, indices) * y_indicators
    return [x_grad, None, y_grad]