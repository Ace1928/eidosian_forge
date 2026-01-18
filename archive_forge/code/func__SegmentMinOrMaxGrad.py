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
def _SegmentMinOrMaxGrad(op, grad):
    """ Gradient for SegmentMin and SegmentMax. """
    zeros = array_ops.zeros_like(op.inputs[0], dtype=op.inputs[0].dtype)
    gathered_outputs = array_ops.gather(op.outputs[0], op.inputs[1])
    is_selected = math_ops.equal(op.inputs[0], gathered_outputs)
    num_selected = math_ops.segment_sum(math_ops.cast(is_selected, grad.dtype), op.inputs[1])
    weighted_grads = math_ops.divide(grad, num_selected)
    gathered_grads = array_ops.gather(weighted_grads, op.inputs[1])
    return (array_ops.where_v2(is_selected, gathered_grads, zeros), None)