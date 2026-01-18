from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as feature_column_v1
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _baseline_logit_fn_builder(num_outputs, weight_column=None):
    """Function builder for a baseline logit_fn.

  Args:
    num_outputs: Number of outputs for the model.
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It will be multiplied by the loss of the example.

  Returns:
    A logit_fn (see below).
  """

    def baseline_logit_fn(features):
        """Baseline model logit_fn.

    The baseline model simply learns a bias, so the output logits are a
    `Variable` with one weight for each output that learns the bias for the
    corresponding output.

    Args:
      features: The first item returned from the `input_fn` passed to `train`,
        `evaluate`, and `predict`. This should be a single `Tensor` or dict with
        `Tensor` values.

    Returns:
      A `Tensor` representing the logits.
    """
        weight_column_key = _get_weight_column_key(weight_column)
        size_checks, batch_size = _get_batch_size_and_size_checks(features, weight_column_key)
        with tf.control_dependencies(size_checks):
            with tf.compat.v1.variable_scope('baseline'):
                bias = tf.compat.v1.get_variable('bias', shape=[num_outputs], initializer=tf.compat.v1.initializers.zeros)
                return tf.math.multiply(bias, tf.ones([batch_size, num_outputs]))
    return baseline_logit_fn