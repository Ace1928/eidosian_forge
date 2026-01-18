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
@ops.RegisterGradient('FusedBatchNormGrad')
def _FusedBatchNormGradGrad(op, *grad):
    """Returns the gradients for the 3 inputs of FusedBatchNormGrad.

  Args:
    op: The FusedBatchNormGradOp for which we need to compute gradients.
    *grad: An argument list for tensors of gradients wrt the outputs with
      grad[0] as grad_grad_x, grad[1] as grad_grad_scale, grad[2] as
      grad_grad_offset.

  Returns:
    A tuple (grad_grad_y, grad_x, grad_scale, None, None), where grad_grad_y
    is the gradient for grad_y, grad_x the gradient for x, grad_scale the
    gradient for scale.
  """
    data_format = op.get_attr('data_format')
    epsilon = op.get_attr('epsilon')
    is_training = op.get_attr('is_training')
    grad_y = op.inputs[0]
    x = op.inputs[1]
    scale = op.inputs[2]
    pop_mean = op.inputs[3]
    pop_var = op.inputs[4]
    grad_grad_x = grad[0]
    grad_grad_scale = grad[1]
    grad_grad_offset = grad[2]
    with backprop.GradientTape() as tape:
        tape.watch(grad_y)
        tape.watch(x)
        tape.watch(scale)
        grad_x, grad_scale, grad_offset = _BatchNormGrad(grad_y, x, scale, pop_mean, pop_var, epsilon, data_format, is_training)
        grad_initial = [grad_grad_x, grad_grad_scale, grad_grad_offset]
    grad_grad_y, grad_x, grad_scale = tape.gradient([grad_x, grad_scale, grad_offset], [grad_y, x, scale], grad_initial)
    return (grad_grad_y, grad_x, grad_scale, None, None)