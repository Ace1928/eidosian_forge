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
@ops.RegisterGradient('TanhGrad')
def _TanhGradGrad(op, grad):
    with ops.control_dependencies([grad]):
        a = math_ops.conj(op.inputs[0])
        b = math_ops.conj(op.inputs[1])
        return (grad * -2.0 * b * a, gen_math_ops.tanh_grad(a, grad))