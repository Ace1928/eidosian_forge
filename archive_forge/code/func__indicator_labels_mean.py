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
def _indicator_labels_mean(labels, weights=None, name=None):
    with ops.name_scope(name, 'labels_mean', (labels, weights)) as scope:
        labels = tf.cast(labels, name='labels', dtype=tf.dtypes.float32)
        if weights is not None:
            weights = tf.compat.v2.__internal__.ops.broadcast_weights(weights, labels)
        return tf.compat.v1.metrics.mean(labels, weights=weights, name=scope)