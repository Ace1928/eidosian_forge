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
def _DepthwiseConv2dNumpy(x1, x2, strides, padding, data_format, dilations):
    """Compute depthwise_conv2d using Numpy.

  This allows use to test TensorFlow's depthwise_conv2d by comparing to the
  Numpy version.

  Unlike `_DepthwiseConv2dNumpyBasic`, this supports more advanced features
  like padding.

  Args:
    x1: The input Numpy array.
    x2: The filter Numpy array.
    strides: A Python list of 4 elements representing the strides.
    padding: The padding. "SAME", "VALID", or a list of explicit paddings.
    data_format: "NHWC" or "NCHW".
    dilations: A list of 2 elements, representing the dilations.

  Returns:
    The depthwise conv2d as a Numpy array.
  """
    if data_format == 'NCHW':
        x1 = np.transpose(x1, (0, 3, 1, 2))
        strides = [strides[0], strides[3], strides[1], strides[2]]
        if dilations:
            dilations = [dilations[0], dilations[3], dilations[1], dilations[2]]
    if dilations:
        fh, fw, c, o = x2.shape
        new_fh = (fh - 1) * dilations[0] + 1
        new_fw = (fw - 1) * dilations[1] + 1
        new_x2 = np.zeros((new_fh, new_fw, c, o))
        for i in range(fh):
            for j in range(fw):
                new_x2[i * dilations[0], j * dilations[1], :] = x2[i, j, :, :]
        x2 = new_x2
    if padding == 'SAME':

        def PaddingsForDim(input_dim, filter_dim, stride):
            """Computes paddings for a single dimension."""
            if input_dim % stride == 0:
                total_padding = max(filter_dim - stride, 0)
            else:
                total_padding = max(filter_dim - input_dim % stride, 0)
            pad_before = total_padding // 2
            pad_after = total_padding - pad_before
            return (pad_before, pad_after)
        padding = [(0, 0), PaddingsForDim(x1.shape[1], x2.shape[0], strides[1]), PaddingsForDim(x1.shape[2], x2.shape[1], strides[2]), (0, 0)]
    elif padding == 'VALID':
        padding = [(0, 0)] * 4
    x1 = np.pad(x1, padding, 'constant')
    y = _DepthwiseConv2dNumpyBasic(x1, x2, strides)
    if data_format == 'NCHW':
        y = np.transpose(y, (0, 2, 3, 1))
    return y