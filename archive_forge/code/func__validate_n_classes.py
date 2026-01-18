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
def _validate_n_classes(n_classes):
    """Validates n_classes argument.

  Required arguments: n_classes.

  Args:
    n_classes: The number of classes.

  Raises:
    ValueError: If n_classes is <= 2 and n_classes is a Python integer.
  Returns:
    n_classes in its original type.
  """
    if isinstance(n_classes, int) and n_classes <= 2:
        raise ValueError('n_classes must be > 2: %s.' % n_classes)
    n_classes_as_tensor = ops.convert_to_tensor(n_classes)
    assert_n_classes = tf.compat.v1.debugging.assert_greater(n_classes_as_tensor, 2, message='n_classes must be greater than 2')
    with tf.control_dependencies([assert_n_classes]):
        tf.no_op()
    return n_classes