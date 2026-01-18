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
def _validate_estimator_spec_train_op(train_op, mode):
    """Validate train_op inputs for EstimatorSpec or TPUEstimatorSpec.

  Args:
    train_op: Op for the training step.
    mode: A `ModeKeys`. Used to determine whether the train_op is acceptable for
      use in the current mode; for example, if we are not training, this can be
      None.

  Returns:
    train_op: Op for the training step.

  Raises:
    ValueError: If no train_op is passed during training.
    TypeError:  If:
                - train_op is neither a `Tensor` nor an Op.
                - train_op is not part of the default graph.
  """
    if train_op is None:
        if mode == ModeKeys.TRAIN:
            raise ValueError('Missing train_op.')
    else:
        default_graph = tf.compat.v1.get_default_graph()
        _check_is_tensor_or_operation(train_op, 'train_op')
        if isinstance(train_op, tf.Variable):
            train_op = train_op.op
        if not (tf.executing_eagerly() or train_op.graph is default_graph):
            raise ValueError(_default_graph_error_message_template.format('train_op', train_op.name))
    return train_op