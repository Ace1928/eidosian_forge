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
@ops.RegisterGradient('EuclideanNorm')
def _EuclideanNormGrad(op, grad):
    """Gradient for EuclideanNorm."""
    output = op.outputs[0]
    if not op.get_attr('keep_dims'):
        output_shape_kept_dims = math_ops.reduced_shape(array_ops.shape(op.inputs[0]), op.inputs[1])
        output = array_ops.reshape(output, output_shape_kept_dims)
        grad = array_ops.reshape(grad, output_shape_kept_dims)
    return (math_ops.truediv(op.inputs[0], output / grad), None)