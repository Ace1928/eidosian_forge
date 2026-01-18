from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column import feature_column_v2 as fc_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils import sdca_ops
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _compute_fraction_of_zero(variables):
    """Given a linear variables list, compute the fraction of zero weights.

  Args:
    variables: A list or list of list of variables

  Returns:
    The fraction of zeros (sparsity) in the linear model.
  """
    with ops.name_scope('zero_fraction'):
        variables = tf.nest.flatten(variables)
        with ops.name_scope('total_size'):
            sizes = [tf.compat.v1.size(x, out_type=tf.dtypes.int64) for x in variables]
            total_size_int64 = tf.math.add_n(sizes)
        with ops.name_scope('total_zero'):
            total_zero_float32 = tf.math.add_n([tf.compat.v1.cond(tf.math.equal(size, tf.constant(0, dtype=tf.dtypes.int64)), true_fn=lambda: tf.constant(0, dtype=tf.dtypes.float32), false_fn=lambda: tf.math.zero_fraction(x) * tf.cast(size, dtype=tf.dtypes.float32), name='zero_count') for x, size in zip(variables, sizes)])
        with ops.name_scope('compute'):
            total_size_float32 = tf.cast(total_size_int64, dtype=tf.dtypes.float32, name='float32_size')
            zero_fraction_or_nan = total_zero_float32 / total_size_float32
        zero_fraction_or_nan = tf.identity(zero_fraction_or_nan, name='zero_fraction_or_nan')
        return zero_fraction_or_nan