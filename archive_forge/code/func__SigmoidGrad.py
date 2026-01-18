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
@ops.RegisterGradient('Sigmoid')
def _SigmoidGrad(op, grad):
    """Returns grad * sigmoid(x) * (1 - sigmoid(x))."""
    y = op.outputs[0]
    with ops.control_dependencies([grad]):
        y = math_ops.conj(y)
        return gen_math_ops.sigmoid_grad(y, grad)