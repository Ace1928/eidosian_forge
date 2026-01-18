import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def _testBias(self, np_inputs, np_bias, use_gpu=False):
    np_val = self._npBias(np_inputs, np_bias)
    with self.cached_session(use_gpu=use_gpu):
        tf_val = self.evaluate(nn_ops.bias_add(np_inputs, np_bias))
    self.assertAllCloseAccordingToType(np_val, tf_val)