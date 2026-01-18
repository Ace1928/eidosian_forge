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
def _check_logits_final_dim(logits, expected_logits_dimension):
    """Checks that logits shape is [D0, D1, ... DN, logits_dimension]."""
    with ops.name_scope(None, 'logits', (logits,)) as scope:
        logits = tf.cast(logits, dtype=tf.dtypes.float32)
        logits_shape = tf.compat.v1.shape(logits)
        assert_rank = tf.compat.v1.debugging.assert_rank_at_least(logits, 2, data=[logits_shape], message='logits shape must be [D0, D1, ... DN, logits_dimension]')
        with tf.control_dependencies([assert_rank]):
            static_shape = logits.shape
            if static_shape.ndims is not None and static_shape[-1] is not None:
                if isinstance(expected_logits_dimension, int) and static_shape[-1] != expected_logits_dimension:
                    raise ValueError('logits shape must be [D0, D1, ... DN, logits_dimension=%s], got %s.' % (expected_logits_dimension, static_shape))
                return logits
            assert_dimension = tf.compat.v1.debugging.assert_equal(expected_logits_dimension, logits_shape[-1], data=[logits_shape], message='logits shape must be [D0, D1, ... DN, logits_dimension=%s]' % (expected_logits_dimension,))
            with tf.control_dependencies([assert_dimension]):
                return tf.identity(logits, name=scope)