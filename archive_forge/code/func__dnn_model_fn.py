from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _dnn_model_fn(features, labels, mode, head, hidden_units, feature_columns, optimizer='Adagrad', activation_fn=tf.nn.relu, dropout=None, input_layer_partitioner=None, config=None, use_tpu=False, batch_norm=False):
    """Deep Neural Net model_fn v1.

  Args:
    features: dict of `Tensor`.
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
      `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction. See
      `ModeKeys`.
    head: A `head_lib._Head` instance.
    hidden_units: Iterable of integer number of hidden units per layer.
    feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
    optimizer: String, `tf.Optimizer` object, or callable that creates the
      optimizer to use for training. If not specified, will use the Adagrad
      optimizer with a default learning rate of 0.05.
    activation_fn: Activation function applied to each layer.
    dropout: When not `None`, the probability we will drop out a given
      coordinate.
    input_layer_partitioner: Partitioner for input layer. Defaults to
      `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
    config: `RunConfig` object to configure the runtime settings.
    use_tpu: Whether to make a DNN model able to run on TPU. Will make function
      return a `_TPUEstimatorSpec` instance and disable variable partitioning.
    batch_norm: Whether to use batch normalization after each hidden layer.

  Returns:
    An `EstimatorSpec` instance.

  Raises:
    ValueError: If features has the wrong type.
  """
    optimizer = optimizers.get_optimizer_instance(optimizer, learning_rate=_LEARNING_RATE)
    _validate_features(features)
    num_ps_replicas = config.num_ps_replicas if config else 0
    partitioner = None if use_tpu else tf.compat.v1.min_max_variable_partitioner(max_partitions=num_ps_replicas)
    with tf.compat.v1.variable_scope('dnn', values=tuple(six.itervalues(features)), partitioner=partitioner):
        input_layer_partitioner = input_layer_partitioner or (None if use_tpu else tf.compat.v1.min_max_variable_partitioner(max_partitions=num_ps_replicas, min_slice_size=64 << 20))
        logit_fn = dnn_logit_fn_builder(units=head.logits_dimension, hidden_units=hidden_units, feature_columns=feature_columns, activation_fn=activation_fn, dropout=dropout, input_layer_partitioner=input_layer_partitioner, batch_norm=batch_norm)
        logits = logit_fn(features=features, mode=mode)
        return _get_dnn_estimator_spec(use_tpu, head, features, labels, mode, logits, optimizer)