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
def _testScalarHandling(self, expected_regex):
    with ops_lib.Graph().as_default(), self.session(use_gpu=False) as sess:
        with self.assertRaisesRegex(errors_impl.InvalidArgumentError, expected_regex):
            labels = array_ops.placeholder(dtypes.int32, shape=[None, 1])
            logits = array_ops.placeholder(dtypes.float32, shape=[None, 3])
            ce = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=array_ops.squeeze(labels), logits=logits)
            labels_v2 = np.zeros((1, 1), dtype=np.int32)
            logits_v2 = np.random.randn(1, 3)
            sess.run([ce], feed_dict={labels: labels_v2, logits: logits_v2})