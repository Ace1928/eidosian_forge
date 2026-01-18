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
@ops.RegisterGradient('Complex')
def _ComplexGrad(op, grad):
    """Returns the real and imaginary components of 'grad', respectively."""
    x = op.inputs[0]
    y = op.inputs[1]
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
    return (array_ops.reshape(math_ops.reduce_sum(math_ops.real(grad), rx), sx), array_ops.reshape(math_ops.reduce_sum(math_ops.imag(grad), ry), sy))