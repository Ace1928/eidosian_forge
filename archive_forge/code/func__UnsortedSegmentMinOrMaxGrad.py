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
def _UnsortedSegmentMinOrMaxGrad(op, grad):
    """ Gradient for UnsortedSegmentMin and UnsortedSegmentMax. """
    gathered_outputs, zero_clipped_indices, is_positive = _GatherDropNegatives(op.outputs[0], op.inputs[1])
    is_selected = math_ops.equal(op.inputs[0], gathered_outputs)
    is_selected = math_ops.logical_and(is_selected, is_positive)
    num_selected = math_ops.unsorted_segment_sum(math_ops.cast(is_selected, grad.dtype), op.inputs[1], op.inputs[2])
    weighted_grads = math_ops.divide(grad, num_selected)
    gathered_grads, _, _ = _GatherDropNegatives(weighted_grads, None, zero_clipped_indices, is_positive)
    zeros = array_ops.zeros_like(gathered_grads)
    return (array_ops.where_v2(is_selected, gathered_grads, zeros), None, None)