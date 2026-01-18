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
@ops.RegisterGradient('FloorMod')
def _FloorModGrad(op, grad):
    """Returns grad * (1, -floor(x/y))."""
    x = math_ops.conj(op.inputs[0])
    y = math_ops.conj(op.inputs[1])
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
    floor_xy = math_ops.floor_div(x, y)
    gx = array_ops.reshape(math_ops.reduce_sum(grad, rx), sx)
    gy = array_ops.reshape(math_ops.reduce_sum(grad * math_ops.negative(floor_xy), ry), sy)
    return (gx, gy)