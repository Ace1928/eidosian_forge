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
def _MaximumMinimumGradInputOnly(op, grad, selector_op):
    x = op.inputs[0]
    y = op.inputs[1]
    zeros = array_ops.zeros_like(grad)
    xmask = selector_op(x, y)
    xgrad = array_ops.where_v2(xmask, grad, zeros)
    ygrad = None
    return (xgrad, ygrad)