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
def _npXent(self, labels, logits):
    logits = np.reshape(logits, [-1, logits.shape[-1]])
    labels = np.reshape(labels, [-1])
    batch_dim = 0
    class_dim = 1
    batch_size = logits.shape[batch_dim]
    e = np.exp(logits - np.reshape(np.amax(logits, axis=class_dim), [batch_size, 1]))
    probs = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    labels_mat = np.zeros_like(probs).astype(probs.dtype)
    labels_mat[np.arange(batch_size), labels] = 1.0
    gradient = probs - labels_mat
    loss = -np.sum(labels_mat * np.log(probs + 1e-20), axis=1)
    return (loss, gradient)