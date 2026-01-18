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
@ops.RegisterGradient('NthElement')
def _NthElementGrad(op, grad):
    """Return the gradients for NthElement.

  Args:
    op: The NthElementOp for which we need to generate gradients.
    grad: Tensor. The gradients passed to the NthElementOp

  Returns:
    A list of two tensors, the first being the gradient w.r.t. the input,
    the second being the gradient w.r.t. the N (None).
  """
    input = op.inputs[0]
    output = op.outputs[0]
    indicators = math_ops.cast(math_ops.equal(array_ops.expand_dims(output, -1), input), grad.dtype)
    grad = array_ops.expand_dims(grad, -1)
    num_selected = array_ops.expand_dims(math_ops.reduce_sum(indicators, -1), -1)
    return [math_ops.divide(indicators, num_selected) * grad, None]