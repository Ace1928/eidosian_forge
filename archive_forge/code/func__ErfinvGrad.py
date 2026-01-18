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
@ops.RegisterGradient('Erfinv')
def _ErfinvGrad(op, grad):
    """Returns grad * sqrt(pi) / 2 * exp(erfinv(x)**2)."""
    root_pi_over_two = constant_op.constant(np.sqrt(np.pi) / 2, dtype=grad.dtype)
    with ops.control_dependencies([grad]):
        return grad * root_pi_over_two * math_ops.exp(math_ops.square(op.outputs[0]))