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
def _baseline_model_fn(features, labels, mode, head, optimizer, weight_column=None, config=None):
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
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It will be multiplied by the loss of the example.
    config: `RunConfig` object to configure the runtime settings.

  Raises:
    KeyError: If weight column is specified but not present.
    ValueError: If features is an empty dictionary.

  Returns:
    An `EstimatorSpec` instance.
  """
    del config
    logit_fn = _baseline_logit_fn_builder(head.logits_dimension, weight_column)
    logits = logit_fn(features)

    def train_op_fn(loss):
        opt = optimizers.get_optimizer_instance(optimizer, learning_rate=_LEARNING_RATE)
        return opt.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    return head.create_estimator_spec(features=features, mode=mode, logits=logits, labels=labels, train_op_fn=train_op_fn)