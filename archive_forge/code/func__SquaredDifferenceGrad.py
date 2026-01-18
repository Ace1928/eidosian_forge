import numpy as np
from tensorflow.python.compat import compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import special_math_ops
@ops.RegisterGradient('SquaredDifference')
def _SquaredDifferenceGrad(op, grad):
    """Returns the gradient for (x-y)^2."""
    x = op.inputs[0]
    y = op.inputs[1]
    skip_input_indices = None
    try:
        skip_input_indices = op.skip_input_indices
    except AttributeError:
        pass
    with ops.control_dependencies([grad]):
        x_grad = math_ops.scalar_mul(2.0, grad) * (x - y)
    if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(x, y, grad):
        return (x_grad, -x_grad)
    (sx, rx, must_reduce_x), (sy, ry, must_reduce_y) = SmartBroadcastGradientArgs(x, y, grad)
    if skip_input_indices is not None and 0 in skip_input_indices:
        gx = None
    elif must_reduce_x:
        gx = array_ops.reshape(math_ops.reduce_sum(x_grad, rx), sx)
    else:
        gx = x_grad
    if skip_input_indices is not None and 1 in skip_input_indices:
        gy = None
    elif must_reduce_y:
        gy = -array_ops.reshape(math_ops.reduce_sum(x_grad, ry), sy)
    else:
        gy = -x_grad
    return (gx, gy)