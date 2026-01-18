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
def _MaximumMinimumGrad(op, grad, selector_op):
    """Factor out the code for the gradient of Maximum or Minimum."""
    y = op.inputs[1]
    skip_input_indices = None
    try:
        skip_input_indices = op.skip_input_indices
        if skip_input_indices is not None and 1 in skip_input_indices and _IsScalar(y):
            return _MaximumMinimumGradInputOnly(op, grad, selector_op)
    except AttributeError:
        pass
    x = op.inputs[0]
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    zeros = array_ops.zeros_like(grad)
    xmask = selector_op(x, y)
    rx, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
    if skip_input_indices is not None and 0 in skip_input_indices:
        gx = None
    else:
        xgrad = array_ops.where_v2(xmask, grad, zeros)
        gx = array_ops.reshape(math_ops.reduce_sum(xgrad, rx), sx)
    if skip_input_indices is not None and 1 in skip_input_indices:
        gy = None
    else:
        ygrad = array_ops.where_v2(xmask, zeros, grad)
        gy = array_ops.reshape(math_ops.reduce_sum(ygrad, ry), sy)
    return (gx, gy)