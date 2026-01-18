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
def _FusedMulAddSub2Grad(op, grad, sgn):
    x1 = op.inputs[0]
    y1 = op.inputs[1]
    x2 = op.inputs[2]
    y2 = op.inputs[3]
    (sx1, rx1, must_reduce_x1), _ = SmartBroadcastGradientArgs(x1, grad, grad)
    (sy1, ry1, must_reduce_y1), _ = SmartBroadcastGradientArgs(y1, grad, grad)
    (sx2, rx2, must_reduce_x2), _ = SmartBroadcastGradientArgs(x2, grad, grad)
    (sy2, ry2, must_reduce_y2), _ = SmartBroadcastGradientArgs(y2, grad, grad)
    gx1 = gen_math_ops.mul(grad, y1)
    gy1 = gen_math_ops.mul(grad, x1)
    gx2 = gen_math_ops.mul(grad, y2) * sgn
    gy2 = gen_math_ops.mul(grad, x2) * sgn
    return [gx1 if not must_reduce_x1 else math_ops.reduce_sum(gx1, rx1), gy1 if not must_reduce_y1 else math_ops.reduce_sum(gy1, ry1), gx2 if not must_reduce_x2 else math_ops.reduce_sum(gx2, rx2), gy2 if not must_reduce_y2 else math_ops.reduce_sum(gy2, ry2)]