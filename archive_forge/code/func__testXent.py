import numpy as np
from tensorflow.python.eager import backprop as backprop_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def _testXent(self, np_labels, np_logits):
    np_loss, np_gradient = self._npXent(labels=np_labels, logits=np_logits)
    tf_loss, tf_gradient = self._opFwdBwd(labels=np_labels, logits=np_logits)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)