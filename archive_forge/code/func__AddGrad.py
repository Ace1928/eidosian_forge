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
@ops.RegisterGradient('Add')
@ops.RegisterGradient('AddV2')
def _AddGrad(op, grad):
    """Gradient for Add."""
    y = op.inputs[1]
    skip_input_indices = None
    try:
        skip_input_indices = op.skip_input_indices
        if skip_input_indices is not None and 1 in skip_input_indices and _IsScalar(y):
            return (grad, None)
    except AttributeError:
        pass
    x = op.inputs[0]
    if isinstance(grad, tensor.Tensor) and _ShapesFullySpecifiedAndEqual(x, y, grad):
        return (grad, grad)
    (sx, rx, must_reduce_x), (sy, ry, must_reduce_y) = SmartBroadcastGradientArgs(x, y, grad)
    if skip_input_indices is not None and 0 in skip_input_indices:
        gx = None
    elif not must_reduce_x:
        gx = grad
    else:
        gx = array_ops.reshape(math_ops.reduce_sum(grad, rx), sx)
    if skip_input_indices is not None and 1 in skip_input_indices:
        gy = None
    elif not must_reduce_y:
        gy = grad
    else:
        gy = array_ops.reshape(math_ops.reduce_sum(grad, ry), sy)
    return (gx, gy)