import functools
import itertools
import operator
from tensorflow.python.eager import backprop
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
@ops.RegisterGradient('BiasAddGrad')
def _BiasAddGradGrad(op, received_grad):
    """Gradient for the BiasAddGrad op.

  Args:
    op: BiasAddGrad op for which we are calculating gradients.
    received_grad: The gradients passed to the BiasAddGrad op.

  Returns:
    A single gradient Tensor for the input to BiasAddGrad (which
    is the gradient of the bias term in BiasAdd)
  """
    try:
        data_format = op.get_attr('data_format')
    except ValueError:
        data_format = None
    shape = array_ops.shape(op.inputs[0])
    bias_shape = array_ops.shape(received_grad)
    if data_format == b'NCHW':
        expanded_shape = array_ops.concat([array_ops.ones_like(shape[:1]), bias_shape, array_ops.ones_like(shape[2:])], 0)
        tile_mults = array_ops.concat([shape[:1], [1], shape[2:]], 0)
    else:
        expanded_shape = array_ops.concat([array_ops.ones_like(shape[:-1]), bias_shape], 0)
        tile_mults = array_ops.concat([shape[:-1], [1]], 0)
    expanded_grad = array_ops.reshape(received_grad, expanded_shape)
    return array_ops.tile(expanded_grad, tile_mults)