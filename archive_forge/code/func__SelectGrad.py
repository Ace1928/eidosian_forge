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
@ops.RegisterGradient('Select')
def _SelectGrad(op, grad):
    c = op.inputs[0]
    x = op.inputs[1]
    zeros = array_ops.zeros_like(x)
    return (None, array_ops.where(c, grad, zeros), array_ops.where(c, zeros, grad))