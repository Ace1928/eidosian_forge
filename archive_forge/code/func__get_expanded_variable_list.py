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
def _get_expanded_variable_list(var_list):
    """Given an iterable of variables, expands them if they are partitioned.

  Args:
    var_list: An iterable of variables.

  Returns:
    A list of variables where each partitioned variable is expanded to its
    components.
  """
    returned_list = []
    for variable in var_list:
        if isinstance(variable, tf.Variable) or tf.compat.v2.__internal__.ops.is_resource_variable(variable) or isinstance(variable, tf.Tensor):
            returned_list.append(variable)
        else:
            returned_list.extend(list(variable))
    return returned_list