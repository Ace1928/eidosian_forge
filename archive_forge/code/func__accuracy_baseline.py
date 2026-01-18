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
def _accuracy_baseline(labels_mean):
    """Return accuracy baseline based on labels mean.

  This is the best the model could do by always predicting one class.

  Args:
    labels_mean: Tuple of value and update op.

  Returns:
    Tuple of value and update op.
  """
    with ops.name_scope(None, 'accuracy_baseline', labels_mean):
        value, update_op = labels_mean
        return (tf.math.maximum(value, 1.0 - value, name='value'), tf.math.maximum(update_op, 1 - update_op, name='update_op'))