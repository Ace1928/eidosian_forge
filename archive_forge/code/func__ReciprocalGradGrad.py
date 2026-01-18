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
@ops.RegisterGradient('ReciprocalGrad')
def _ReciprocalGradGrad(op, grad):
    b = op.inputs[1]
    with ops.control_dependencies([grad]):
        ca = math_ops.conj(op.inputs[0])
        cg = math_ops.conj(grad)
        return (cg * -2.0 * b * ca, gen_math_ops.reciprocal_grad(ca, grad))