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
@ops.RegisterGradient('BesselI0')
def _BesselI0Grad(op, grad):
    """Compute gradient of bessel_i0(x) with respect to its argument."""
    x = op.inputs[0]
    with ops.control_dependencies([grad]):
        partial_x = special_math_ops.bessel_i1(x)
        return grad * partial_x