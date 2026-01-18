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
@ops.RegisterGradient('BesselK1e')
def _BesselK1eGrad(op, grad):
    """Compute gradient of bessel_k1e(x) with respect to its argument."""
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        partial_x = y * (1.0 - math_ops.reciprocal(x)) - special_math_ops.bessel_k0e(x)
        return grad * partial_x