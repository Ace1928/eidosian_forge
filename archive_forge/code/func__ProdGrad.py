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
@ops.RegisterGradient('Prod')
def _ProdGrad(op, grad):
    """Gradient for Prod."""
    input_shape = array_ops.shape(op.inputs[0])
    reduction_indices = array_ops.reshape(op.inputs[1], [-1])
    if not op.get_attr('keep_dims'):
        output_shape_kept_dims = math_ops.reduced_shape(input_shape, op.inputs[1])
        grad = array_ops.reshape(grad, output_shape_kept_dims)
    grad = array_ops.broadcast_to(grad, input_shape)
    with ops.device('/cpu:0'):
        rank = array_ops.rank(op.inputs[0])
        reduction_indices = (reduction_indices + rank) % rank
        reduced = math_ops.cast(reduction_indices, dtypes.int32)
        idx = math_ops.range(0, rank)
        other, _ = gen_array_ops.list_diff(idx, reduced, dtypes.int32)
        perm = array_ops.concat([reduced, other], 0)
        reduced_num = math_ops.reduce_prod(array_ops.gather(input_shape, reduced))
        other_num = math_ops.reduce_prod(array_ops.gather(input_shape, other))
    permuted = array_ops.transpose(op.inputs[0], perm)
    permuted_shape = array_ops.shape(permuted)
    reshaped = array_ops.reshape(permuted, (reduced_num, other_num))
    left = math_ops.cumprod(reshaped, axis=0, exclusive=True)
    right = math_ops.cumprod(reshaped, axis=0, exclusive=True, reverse=True)
    y = array_ops.reshape(math_ops.conj(left) * math_ops.conj(right), permuted_shape)
    out = grad * array_ops.transpose(y, array_ops.invert_permutation(perm))
    return (array_ops.reshape(out, input_shape), None)