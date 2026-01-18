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
def _all_classes(logits, n_classes, label_vocabulary=None):
    batch_size = tf.compat.v1.shape(logits)[0]
    if label_vocabulary:
        classes_list = label_vocabulary
    else:
        classes_list = string_ops.as_string(tf.range(n_classes))
    return tf.tile(input=tf.compat.v1.expand_dims(input=classes_list, axis=0), multiples=[batch_size, 1])