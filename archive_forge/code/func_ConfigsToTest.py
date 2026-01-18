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
def ConfigsToTest():
    """Iterator for different convolution shapes, strides and paddings.

  Returns:
    List of tuples (input_size, filter_size, out_size, stride, padding,
    dilations), the depthwise convolution parameters.
  """

    def Config(input_size, filter_size, out_size, stride=1, padding='SAME', dilations=None):
        return (input_size, filter_size, out_size, stride, padding, dilations)
    return [Config([4, 5, 5, 48], [1, 1, 48, 2], [4, 5, 5, 96]), Config([4, 8, 8, 84], [1, 3, 84, 1], [4, 8, 8, 84]), Config([4, 17, 17, 48], [3, 1, 48, 4], [4, 17, 17, 192]), Config([4, 9, 27, 8], [3, 3, 8, 1], [4, 9, 27, 8]), Config([4, 31, 31, 7], [3, 3, 7, 1], [4, 31, 31, 7]), Config([4, 35, 35, 2], [5, 5, 2, 1], [4, 35, 35, 2]), Config([4, 147, 147, 2], [3, 3, 2, 8], [4, 49, 49, 16], 3, padding='VALID'), Config([3, 299, 299, 3], [3, 2, 3, 8], [3, 150, 150, 24], 2), Config([5, 183, 183, 1], [5, 5, 1, 2], [5, 92, 92, 2], 2), Config([5, 183, 183, 1], [5, 5, 1, 2], [5, 183, 183, 2], dilations=[2, 2]), Config([5, 41, 35, 2], [4, 7, 2, 2], [5, 32, 23, 4], padding='VALID', dilations=[3, 2])]