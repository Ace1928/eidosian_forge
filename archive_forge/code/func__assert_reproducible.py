import collections
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
def _assert_reproducible(self, operation):
    with test_util.force_gpu():
        result_1 = operation()
        result_2 = operation()
    self.assertAllEqual(result_1, result_2)