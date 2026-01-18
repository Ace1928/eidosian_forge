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
@ops.RegisterGradient('Xlogy')
def _XLogyGrad(op, grad):
    """Returns gradient of xlogy(x, y) with respect to x and y."""
    x = op.inputs[0]
    y = op.inputs[1]
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
    with ops.control_dependencies([grad]):
        not_zero_x = math_ops.cast(math_ops.not_equal(x, math_ops.cast(0.0, dtype=x.dtype)), dtype=x.dtype)
        partial_x = gen_math_ops.xlogy(not_zero_x, y)
        partial_y = gen_math_ops.xdivy(x, y)
        return (array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx), array_ops.reshape(math_ops.reduce_sum(partial_y * grad, ry), sy))