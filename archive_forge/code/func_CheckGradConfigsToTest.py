import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
def CheckGradConfigsToTest():
    """Iterator for different convolution shapes, strides and paddings.

  compute_gradient_error() is very expensive. So the configs should be
  relatively small.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

    def Config(input_size, filter_size, out_size, stride=1, padding='SAME', dilations=None):
        return (input_size, filter_size, out_size, stride, padding, dilations)
    return [Config([2, 5, 8, 1], [4, 4, 1, 2], [2, 5, 8, 2]), Config([4, 5, 5, 1], [2, 2, 1, 2], [4, 2, 2, 2], 2, padding='VALID'), Config([2, 4, 4, 2], [3, 1, 2, 2], [2, 4, 4, 4]), Config([1, 15, 15, 2], [1, 3, 2, 1], [1, 15, 15, 2]), Config([2, 15, 16, 1], [3, 3, 1, 2], [2, 5, 5, 2], 3, padding='VALID'), Config([2, 5, 8, 1], [4, 3, 1, 2], [2, 5, 8, 2], dilations=[1, 2]), Config([1, 3, 1, 2], [2, 1, 2, 1], [1, 3, 1, 2]), Config([2, 2, 3, 2], [2, 1, 2, 1], [2, 2, 3, 2]), Config([2, 2, 3, 1], [2, 2, 1, 1], [2, 2, 3, 1])]