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
@ops.RegisterGradient('IsotonicRegression')
def _IsotonicRegressionGrad(op, grad_output, grad_segments):
    """Gradient for the isotonic regression function.

  Args:
    op: The IsotonicRegression tensorflow op.
    grad_output: Tensor of incoming gradients with respect to the output.
    grad_segments: Tensor of incoming gradients with respect to the segments.

  Returns:
    A tensor, same size as `grad_output` with the gradient with respect to
    the input.
  """
    del grad_segments
    segments = op.outputs[1]
    return _MeanAggregator(grad_output, segments)