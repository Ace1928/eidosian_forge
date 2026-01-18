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
@ops.RegisterGradient('Polygamma')
def _PolygammaGrad(op, grad):
    """Returns gradient of psi(n, x) with respect to n and x."""
    n = op.inputs[0]
    x = op.inputs[1]
    sn = array_ops.shape(n)
    sx = array_ops.shape(x)
    unused_rn, rx = gen_array_ops.broadcast_gradient_args(sn, sx)
    with ops.control_dependencies([grad]):
        n = math_ops.conj(n)
        x = math_ops.conj(x)
        partial_x = math_ops.polygamma(n + 1, x)
        return (None, array_ops.reshape(math_ops.reduce_sum(partial_x * grad, rx), sx))