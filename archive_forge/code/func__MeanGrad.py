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
@ops.RegisterGradient('Mean')
def _MeanGrad(op, grad):
    """Gradient for Mean."""
    sum_grad = _SumGrad(op, grad)[0]
    input_shape = op.inputs[0]._shape_tuple()
    output_shape = op.outputs[0]._shape_tuple()
    if input_shape is not None and output_shape is not None and (None not in input_shape) and (None not in output_shape):
        input_size = np.prod(input_shape)
        output_size = np.prod(output_shape)
        factor = input_size // max(output_size, 1)
        factor = constant_op.constant(factor, dtype=sum_grad.dtype)
    else:
        input_shape = array_ops.shape(op.inputs[0])
        input_rank = array_ops.size(input_shape)
        axes = (op.inputs[1] + input_rank) % input_rank
        factor = math_ops.reduce_prod(array_ops.gather(input_shape, axes))
    return (math_ops.truediv(sum_grad, math_ops.cast(factor, sum_grad.dtype)), None)