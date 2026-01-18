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
def _baseline_model_fn_v2(features, labels, mode, head, optimizer, weight_column=None, config=None, loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE):
    """Model_fn for baseline models.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `train`).
    labels: `Tensor` of labels that are compatible with the `Head` instance.
    mode: Defines whether this is training, evaluation or prediction. See
      `ModeKeys`.
    head: A `Head` instance.
    optimizer: String, `tf.Optimizer` object, or callable that creates the
      optimizer to use for training. If not specified, will use `FtrlOptimizer`
      with a default learning rate of 0.3.
    weight_column: A string or a `NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It will be multiplied by the loss of the example.
    config: `RunConfig` object to configure the runtime settings.
    loss_reduction: One of `tf.keras.losses.Reduction` except `NONE`. Describes
      how to reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.

  Raises:
    KeyError: If weight column is specified but not present.
    ValueError: If features is an empty dictionary.

  Returns:
    An `EstimatorSpec` instance.
  """
    del config
    trainable_variables, logits = _baseline_model_fn_builder_v2(features, head.logits_dimension, weight_column)
    if mode == ModeKeys.TRAIN:
        opt = optimizers.get_optimizer_instance_v2(optimizer, learning_rate=_LEARNING_RATE)
        opt.iterations = tf.compat.v1.train.get_or_create_global_step()

    def train_op_fn(loss):
        if loss_reduction == tf.losses.Reduction.SUM_OVER_BATCH_SIZE:
            num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
            if num_replicas > 1:
                loss *= 1.0 / num_replicas
        return opt.get_updates(loss, trainable_variables)[0]
    return head.create_estimator_spec(features=features, mode=mode, logits=logits, labels=labels, train_op_fn=train_op_fn)