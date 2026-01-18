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
@ops.RegisterGradient('Softmax')
def _SoftmaxGrad(op, grad_softmax):
    """The derivative of the softmax nonlinearity.

  We assume that probs is of shape [batch_size * dim]
  The formula for dsoftmax / dx = (diag(softmax) - softmax * softmax').
  This matrix is diagonal minus a rank one matrix, so it is easy to implement
  as follows:

    grad_x = grad_softmax * softmax - sum(grad_softmax * softmax) * softmax

  Args:
     op: the Softmax op.
     grad_softmax:  the tensor representing the gradient w.r.t. the softmax
       output.

  Returns:
     gradient w.r.t the input to the softmax

  """
    softmax = op.outputs[0]
    sum_channels = math_ops.reduce_sum(grad_softmax * softmax, -1, keepdims=True)
    return (grad_softmax - sum_channels) * softmax