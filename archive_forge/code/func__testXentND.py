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
def _testXentND(self, np_labels, np_logits, dim=-1):
    np_loss, _ = self._npXent(np_labels, np_logits, dim=dim)
    loss = nn_ops.softmax_cross_entropy_with_logits(labels=np_labels, logits=np_logits, dim=dim)
    tf_loss = self.evaluate(loss)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)