import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test
def _testXent2D(self, np_labels, np_logits, with_placeholders=False, expected_gradient=None):
    np_loss, np_gradient = self._npXent(labels=np_labels, logits=np_logits)
    if expected_gradient is not None:
        np_gradient = expected_gradient
    with self.cached_session() as sess:
        if with_placeholders:
            logits_placeholder = array_ops.placeholder(np_logits.dtype)
            labels_placeholder = array_ops.placeholder(np_labels.dtype)
            loss, gradient = self._opFwdBwd(labels_placeholder, logits_placeholder)
            tf_loss, tf_gradient = sess.run([loss, gradient], feed_dict={labels_placeholder: np_labels, logits_placeholder: np_logits})
        else:
            loss, gradient = self._opFwdBwd(np_labels, np_logits)
            tf_loss, tf_gradient = self.evaluate([loss, gradient])
    self.assertAllCloseAccordingToType(np_loss, tf_loss, half_rtol=0.01)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)