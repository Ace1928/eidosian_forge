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
@ops.RegisterGradient('SegmentMean')
def _SegmentMeanGrad(op, grad):
    """Gradient for SegmentMean."""
    input_rank = array_ops.rank(op.inputs[0])
    ones_shape = array_ops.concat([array_ops.shape(op.inputs[1]), array_ops.ones(array_ops.expand_dims(input_rank - 1, 0), dtype=dtypes.int32)], 0)
    ones = array_ops.ones(ones_shape, dtype=grad.dtype)
    scaled_grad = math_ops.divide(grad, math_ops.segment_sum(ones, op.inputs[1]))
    return (array_ops.gather(scaled_grad, op.inputs[1]), None)