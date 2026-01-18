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
@ops.RegisterGradient('Pow')
def _PowGrad(op, grad):
    """Returns grad * (y*x^(y-1), z*log(x))."""
    x = op.inputs[0]
    y = op.inputs[1]
    skip_input_indices = None
    try:
        skip_input_indices = op.skip_input_indices
        if skip_input_indices is not None and 1 in skip_input_indices and _IsScalar(y):
            x = math_ops.conj(x)
            y = math_ops.conj(y)
            return (grad * y * math_ops.pow(x, y - 1), None)
    except AttributeError:
        pass
    (sx, rx, must_reduce_x), (sy, ry, must_reduce_y) = SmartBroadcastGradientArgs(x, y, grad)
    x = math_ops.conj(x)
    y = math_ops.conj(y)
    if skip_input_indices is None or 0 not in skip_input_indices:
        gx = grad * y * math_ops.pow(x, y - 1)
        if must_reduce_x:
            gx = array_ops.reshape(math_ops.reduce_sum(gx, rx), sx)
    else:
        gx = None
    if skip_input_indices is None or 1 not in skip_input_indices:
        z = math_ops.conj(op.outputs[0])
        if x.dtype.is_complex:
            mask = math_ops.not_equal(x, 0)
        else:
            mask = x > 0
        safe_x = array_ops.where(mask, x, array_ops.ones_like(x))
        log_x = array_ops.where(mask, math_ops.log(safe_x), array_ops.zeros_like(x))
        gy = grad * z * log_x
        if must_reduce_y:
            gy = array_ops.reshape(math_ops.reduce_sum(gy, ry), sy)
    else:
        gy = None
    return (gx, gy)