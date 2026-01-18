from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.saved_model import model_utils as export_utils
from tensorflow.python.tpu import tensor_tracer
from tensorflow.python.util import function_utils
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _check_is_tensor_or_operation(x, name):
    if not isinstance(x, (tf.Operation, tf.compat.v2.__internal__.types.Tensor)):
        raise TypeError('{} must be Operation or Tensor, given: {}'.format(name, x))