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
def _validate_scaffold(scaffold):
    """Validate scaffold input for EstimatorSpec.

  Args:
    scaffold: A `tf.train.Scaffold` object that can be used to set
      initialization, saver, and more to be used in training.

  Returns:
    scaffold: A `tf.train.Scaffold` object. If no scaffold is provided, then a
      default is generated.

  Raises:
    TypeError: If the scaffold is not of type `monitored_session.Scaffold`
      or None.
  """
    scaffold = scaffold or tf.compat.v1.train.Scaffold()
    if not isinstance(scaffold, tf.compat.v1.train.Scaffold):
        raise TypeError('scaffold must be tf.train.Scaffold. Given: {}'.format(scaffold))
    return scaffold