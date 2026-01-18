from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _append_update_ops(train_op):
    """Returns `train_op` appending `UPDATE_OPS` collection if present."""
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    if update_ops:
        return tf.group(train_op, *update_ops)
    return train_op