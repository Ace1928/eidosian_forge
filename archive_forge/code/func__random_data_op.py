import collections
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
def _random_data_op(self, shape):
    return constant_op.constant(2 * np.random.random_sample(shape) - 1, dtype=dtypes.float32)