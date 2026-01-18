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
@ops.RegisterGradient('Cumsum')
def _CumsumGrad(op, grad):
    axis = op.inputs[1]
    exclusive = op.get_attr('exclusive')
    reverse = op.get_attr('reverse')
    return [math_ops.cumsum(grad, axis, exclusive=exclusive, reverse=not reverse), None]