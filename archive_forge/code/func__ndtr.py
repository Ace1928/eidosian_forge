import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def _ndtr(x):
    """Implements ndtr core logic."""
    half_sqrt_2 = constant_op.constant(0.5 * np.sqrt(2.0), dtype=x.dtype, name='half_sqrt_2')
    w = x * half_sqrt_2
    z = math_ops.abs(w)
    y = array_ops.where_v2(math_ops.less(z, half_sqrt_2), 1.0 + math_ops.erf(w), array_ops.where_v2(math_ops.greater(w, 0.0), 2.0 - math_ops.erfc(z), math_ops.erfc(z)))
    return 0.5 * y