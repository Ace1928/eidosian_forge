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
@ops.RegisterGradient('Real')
def _RealGrad(_, grad):
    """Returns 'grad' as the real part and set the imaginary part 0."""
    zero = constant_op.constant(0, dtype=grad.dtype)
    return math_ops.complex(grad, zero)