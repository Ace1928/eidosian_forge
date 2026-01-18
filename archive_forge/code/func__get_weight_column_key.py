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
def _get_weight_column_key(weight_column):
    if weight_column is None:
        return None
    if isinstance(weight_column, six.string_types):
        return weight_column
    if not isinstance(weight_column, feature_column_v1._NumericColumn):
        raise TypeError('Weight column must be either a string or _NumericColumn. Given type: {}.'.format(type(weight_column)))
    return weight_column.key()