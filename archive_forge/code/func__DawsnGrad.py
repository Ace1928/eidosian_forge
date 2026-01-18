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
@ops.RegisterGradient('Dawsn')
def _DawsnGrad(op, grad):
    """Compute gradient of dawsn(x) with respect to its argument."""
    x = op.inputs[0]
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        return grad * (1.0 - 2 * x * y)