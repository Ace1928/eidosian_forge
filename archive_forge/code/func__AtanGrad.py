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
@ops.RegisterGradient('Atan')
def _AtanGrad(op, grad):
    """Returns grad * 1/ (1 + x^2)."""
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        x2 = math_ops.square(x)
        one = constant_op.constant(1, dtype=grad.dtype)
        inv = math_ops.reciprocal(math_ops.add(one, x2))
        return grad * inv