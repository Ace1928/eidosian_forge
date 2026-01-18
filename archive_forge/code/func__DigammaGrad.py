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
@ops.RegisterGradient('Digamma')
def _DigammaGrad(op, grad):
    """Compute gradient of the digamma function with respect to its argument."""
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        x = math_ops.conj(x)
        partial_x = math_ops.polygamma(array_ops.constant(1, dtype=x.dtype), x)
        return grad * partial_x