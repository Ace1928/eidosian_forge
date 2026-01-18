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
def _testLabelsPlaceholderScalar(self, expected_error_message):
    with ops_lib.Graph().as_default(), self.session():
        labels = array_ops.placeholder(np.int32)
        y = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=labels, logits=[[7.0]])
        with self.assertRaisesOpError(expected_error_message):
            y.eval(feed_dict={labels: 0})