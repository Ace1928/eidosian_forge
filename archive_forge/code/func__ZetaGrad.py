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
@ops.RegisterGradient('Zeta')
def _ZetaGrad(op, grad):
    """Returns gradient of zeta(x, q) with respect to x and q."""
    x = op.inputs[0]
    q = op.inputs[1]
    sx = array_ops.shape(x)
    sq = array_ops.shape(q)
    unused_rx, rq = gen_array_ops.broadcast_gradient_args(sx, sq)
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        q = math_ops.conj(q)
        partial_q = -x * math_ops.zeta(x + 1, q)
        return (None, array_ops.reshape(math_ops.reduce_sum(partial_q * grad, rq), sq))