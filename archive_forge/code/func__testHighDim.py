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
@test_util.run_in_graph_and_eager_modes()
def _testHighDim(self, labels, logits):
    np_loss, np_gradient = self._npXent(labels=np.array(labels), logits=np.array(logits))
    np_loss = np.reshape(np_loss, np.array(labels).shape)
    tf_loss = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
    with backprop_lib.GradientTape() as tape:
        logits = constant_op.constant(logits)
        tape.watch(logits)
        tf_gradient = tape.gradient(nn_ops.sparse_softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits), [logits])[0]
        tf_gradient = array_ops.reshape(tf_gradient, np_gradient.shape)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)