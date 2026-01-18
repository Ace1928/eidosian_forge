from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@ops.RegisterGradient('ClipByValue')
def _clip_by_value_grad(op, grad):
    """Returns grad of clip_by_value."""
    x = op.inputs[0]
    y = op.inputs[1]
    z = op.inputs[2]
    gdtype = grad.dtype
    sx = array_ops.shape(x)
    sy = array_ops.shape(y)
    sz = array_ops.shape(z)
    gradshape = array_ops.shape(grad)
    zeros = array_ops.zeros(gradshape, gdtype)
    xymask = math_ops.less(x, y)
    xzmask = math_ops.greater(x, z)
    _, ry = gen_array_ops.broadcast_gradient_args(sx, sy)
    _, rz = gen_array_ops.broadcast_gradient_args(sx, sz)
    xgrad = array_ops.where(math_ops.logical_or(xymask, xzmask), zeros, grad)
    ygrad = array_ops.where(xymask, grad, zeros)
    zgrad = array_ops.where(xzmask, grad, zeros)
    gy = array_ops.reshape(math_ops.reduce_sum(ygrad, ry), sy)
    gz = array_ops.reshape(math_ops.reduce_sum(zgrad, rz), sz)
    return (xgrad, gy, gz)