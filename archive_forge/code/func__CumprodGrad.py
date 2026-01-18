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
@ops.RegisterGradient('Cumprod')
def _CumprodGrad(op, grad):
    x = op.inputs[0]
    axis = op.inputs[1]
    exclusive = op.get_attr('exclusive')
    reverse = op.get_attr('reverse')
    prod = math_ops.cumprod(x, axis, exclusive=exclusive, reverse=reverse)
    out = math_ops.cumsum(prod * grad, axis, exclusive=exclusive, reverse=not reverse)
    return [math_ops.div_no_nan(out, x), None]