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
def _sdca_model_fn(features, labels, mode, head, feature_columns, optimizer):
    """A model_fn for linear models that use the SDCA optimizer.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape `[batch_size]`.
    mode: Defines whether this is training, evaluation or prediction. See
      `ModeKeys`.
    head: A `Head` instance.
    feature_columns: An iterable containing all the feature columns used by the
      model.
    optimizer: a `LinearSDCA` instance.

  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: mode or params are invalid, or features has the wrong type.
  """
    assert feature_column_lib.is_feature_column_v2(feature_columns)
    if isinstance(head, (binary_class_head.BinaryClassHead, head_lib._BinaryLogisticHeadWithSigmoidCrossEntropyLoss)):
        loss_type = 'logistic_loss'
    elif isinstance(head, (regression_head.RegressionHead, head_lib._RegressionHeadWithMeanSquaredErrorLoss)):
        assert head.logits_dimension == 1
        loss_type = 'squared_loss'
    else:
        raise ValueError('Unsupported head type: {}'.format(head))
    linear_model_name = 'linear_model'
    if isinstance(head, (binary_class_head.BinaryClassHead, regression_head.RegressionHead)):
        linear_model_name = 'linear/linear_model'
    linear_model = LinearModel(feature_columns=feature_columns, units=1, sparse_combiner='sum', name=linear_model_name)
    logits = linear_model(features)
    bias = linear_model.bias
    variables = linear_model.variables
    bias = _get_expanded_variable_list([bias])
    variables = _get_expanded_variable_list(variables)
    variables = [var for var in variables if var not in bias]
    tf.compat.v1.summary.scalar('bias', bias[0][0])
    tf.compat.v1.summary.scalar('fraction_of_zero_weights', _compute_fraction_of_zero(variables))
    if mode == ModeKeys.TRAIN:
        sdca_model, train_op = optimizer.get_train_step(linear_model.layer._state_manager, head._weight_column, loss_type, feature_columns, features, labels, linear_model.bias, tf.compat.v1.train.get_global_step())
        update_weights_hook = _SDCAUpdateWeightsHook(sdca_model, train_op)
        model_fn_ops = head.create_estimator_spec(features=features, mode=mode, labels=labels, train_op_fn=lambda unused_loss_fn: train_op, logits=logits)
        return model_fn_ops._replace(training_chief_hooks=model_fn_ops.training_chief_hooks + (update_weights_hook,))
    else:
        return head.create_estimator_spec(features=features, mode=mode, labels=labels, logits=logits)