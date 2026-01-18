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
@ops.RegisterGradient('Igamma')
def _IgammaGrad(op, grad):
    """Returns gradient of igamma(a, x) with respect to a and x."""
    a = op.inputs[0]
    x = op.inputs[1]
    sa = array_ops.shape(a)
    sx = array_ops.shape(x)
    ra, rx = gen_array_ops.broadcast_gradient_args(sa, sx)
    with ops.control_dependencies([grad]):
        partial_a = gen_math_ops.igamma_grad_a(a, x)
        partial_x = math_ops.exp(-x + (a - 1) * math_ops.log(x) - math_ops.lgamma(a))
        return (array_ops.reshape(math_ops.reduce_sum(partial_a * grad, ra), sa), array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))