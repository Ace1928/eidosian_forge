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
def PaddingsForDim(input_dim, filter_dim, stride):
    """Computes paddings for a single dimension."""
    if input_dim % stride == 0:
        total_padding = max(filter_dim - stride, 0)
    else:
        total_padding = max(filter_dim - input_dim % stride, 0)
    pad_before = total_padding // 2
    pad_after = total_padding - pad_before
    return (pad_before, pad_after)