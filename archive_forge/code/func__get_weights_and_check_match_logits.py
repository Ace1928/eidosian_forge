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
def _get_weights_and_check_match_logits(features, weight_column, logits, allow_per_logit_weights=False):
    """Fetches weights from features and checks that the shape matches logits.

  Consider logits of shape [D0, D1, ... DN, logits_dimension]. Weights shape
  can be either:
  * [D0, D1, ... DN, logits_dimension] if `allow_per_logit_weights=True`.
  * [D0, D1, ... DN, 1]
  * [D0, D1, ... DN]: In this case, weights is reshaped into
    [D0, D1, ... DN, 1] to work with weight broadcasting rules.

  Args:
    features: The features dict that contains weights.
    weight_column: The weight column. If not given, this method returns 1.
    logits: logits Tensor.
    allow_per_logit_weights: Boolean. Whether we allow weights along the logits
      dimension, namely shape `[D0, D1, ... DN, logits_dimension]`.

  Returns:
    Validated and reshaped weights Tensor.
  Raises:
    ValueError: If the weights `Tensor` cannot be cast into float.
  """
    if allow_per_logit_weights:
        err_msg = 'weights shape must be [D0, D1, ... DN], [D0, D1, ... DN, 1] or [D0, D1, ... DN, logits_dimension]'
    else:
        err_msg = 'weights shape must be [D0, D1, ... DN] or [D0, D1, ... DN, 1]'
    with ops.name_scope(None, 'weights', values=tuple(six.itervalues(features)) + (logits,)) as scope:
        if weight_column is None:
            return 1.0
        if isinstance(weight_column, six.string_types):
            weight_column = tf.feature_column.numeric_column(key=weight_column, shape=(1,))
        if not isinstance(weight_column, (tf.compat.v2.__internal__.feature_column.DenseColumn, feature_column._DenseColumn)):
            raise TypeError('Weight column must be either a string or _DenseColumn. Given type: {}.'.format(type(weight_column)))
        weights = weight_column._get_dense_tensor(feature_column._LazyBuilder(features))
        if not (weights.dtype.is_floating or weights.dtype.is_integer):
            raise ValueError('Weight column should be castable to float. Given dtype: {}'.format(weights.dtype))
        weights = tf.cast(weights, name='weights', dtype=tf.dtypes.float32)
        weights_shape = tf.compat.v1.shape(weights, name='weights_shape')
        logits_shape = tf.compat.v1.shape(logits, name='logits_shape')
        if weights.shape.ndims is not None and logits.shape.ndims is not None and (weights.shape.ndims == logits.shape.ndims - 1):
            assert_dimension = tf.compat.v1.debugging.assert_equal(logits_shape[:-1], weights_shape, message=err_msg, data=['logits_shape: ', logits_shape, 'weights_shape: ', weights_shape])
            with tf.control_dependencies([assert_dimension]):
                return tf.compat.v1.expand_dims(weights, -1, name=scope)
        supported_weights_shape = tf.concat([logits_shape[:-1], [1]], axis=0)
        if allow_per_logit_weights:
            condition = tf.math.reduce_any([tf.reduce_all(tf.math.equal(logits_shape, weights_shape)), tf.reduce_all(tf.math.equal(supported_weights_shape, weights_shape))])
            assert_dimension = tf.debugging.Assert(condition=condition, data=[err_msg, 'logits_shape: ', logits_shape, 'weights_shape: ', weights_shape])
        else:
            assert_dimension = tf.compat.v1.debugging.assert_equal(supported_weights_shape, weights_shape, message=err_msg, data=['logits_shape: ', logits_shape, 'weights_shape: ', weights_shape])
        with tf.control_dependencies([assert_dimension]):
            return tf.identity(weights, name=scope)