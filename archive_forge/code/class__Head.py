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
class _Head(object):
    """Interface for the head/top of a model.

  Given logits (or output of a hidden layer), a Head knows how to compute
  predictions, loss, train_op, metrics and export outputs. It is meant to:

  1. Simplify writing model_fn and to make model_fn more configurable
  2. Support wide range of machine learning models. Since most heads can work
     with logits, they can support DNN, RNN, Wide, Wide&Deep,
     Global objectives, Gradient boosted trees and many other types
     of machine learning models.

  Common usage:
  Here is simplified model_fn to build a DNN regression model.
    ```python
    def _my_dnn_model_fn(features, labels, mode, params, config=None):
      # Optionally your callers can pass head to model_fn as a param.
      head = tf.contrib.estimator.regression_head(...)
      inputs = tf.feature_column.input_layer(features, ...)
      hidden_layer0 = tf.layers.dense(
          inputs, units=1000, activation=tf.nn.relu)
      hidden_layer1 = tf.layers.dense(
          hidden_layer0, units=500, activation=tf.nn.relu)
      logits = tf.layers.dense(
          hidden_layer1, units=head.logits_dimension, activation=None)

      return head.create_estimator_spec(
          features=features,
          labels=labels,
          mode=mode,
          logits=logits,
          optimizer=optimizer)
    ```

  There are cases where computing and applying gradients can not be meaningfully
  captured with optimizer or train_op_fn we support (for example, with sync
  optimizer). In such case, you can take the responsibility on your own. Here is
  a common use case,
    ```python
    estimator_spec = head.create_estimator_spec(
        features=features,
        labels=labels,
        mode=mode,
        logits=logits,
        train_op_fn=lambda _: tf.no_op())
    if mode == ModeKeys.TRAIN:
      optimizer = ...
      sync = tf.train.SyncReplicasOptimizer(opt=optimizer, ...)
      update_op = sync.minimize(
          estimator_spec.loss, global_step=tf.get_global_step())
      hooks = [sync.make_session_run_hook(is_chief)]
      ... update train_op and hooks in EstimatorSpec and return
    ```
  """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def name(self):
        """The name of this head.

    Returns:
      A string.
    """
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractproperty
    def logits_dimension(self):
        """Size of the last dimension of the logits `Tensor`.

    Typically, logits is of shape `[batch_size, logits_dimension]`.

    Returns:
      The expected size of the `logits` tensor.
    """
        raise NotImplementedError('Calling an abstract method.')

    @abc.abstractmethod
    def create_loss(self, features, mode, logits, labels):
        """Returns a loss Tensor from provided logits.

    This function is designed to be used by framework developers.  Almost all
    users should use create_estimator_spec(), which calls this internally.
    `mode` and `features` are most likely not used, but some Head
    implementations may require them.

    Args:
      features: Input `dict` of `Tensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` to be used for loss construction.
      labels: Labels `Tensor`, or `dict` of same.

    Returns:
      A LossSpec that contains
      * the scalar `Tensor` representing reduced weighted training loss
      * the `Tensor` representing the unreduced unweighted loss
      * the `Tensor` representing the example weights
      * possibly processed labels (e.g. vocabulary lookup, shape manipulation,
        etc.)

      To be extendable in the future.
    """
        raise NotImplementedError('Calling an abstract method.')

    def create_estimator_spec(self, features, mode, logits, labels=None, optimizer=None, train_op_fn=None, regularization_losses=None):
        """Returns `EstimatorSpec` that a model_fn can return.

    Please note that,
    + All args must be passed via name.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` to be used by the head.
      labels: Labels `Tensor`, or `dict` of same.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns an op
        to optimize the model with the loss in TRAIN mode. Used if `optimizer`
        is `None`. Exactly one of `train_op_fn` and `optimizer` must be set in
        TRAIN mode. None is allowed in other modes. If you want to optimize loss
        yourself you can pass `lambda _: tf.no_op()` and then use
          EstimatorSpec.loss to compute and apply gradients.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      `EstimatorSpec`.
    """
        try:
            tpu_estimator_spec = self._create_tpu_estimator_spec(features, mode, logits, labels, optimizer, train_op_fn, regularization_losses)
            return tpu_estimator_spec.as_estimator_spec()
        except NotImplementedError:
            raise NotImplementedError('Subclasses of _Head must implement `create_estimator_spec()` or _create_tpu_estimator_spec().')

    def _create_tpu_estimator_spec(self, features, mode, logits, labels=None, optimizer=None, train_op_fn=None, regularization_losses=None):
        """Returns `model_fn._TPUEstimatorSpec` that a model_fn can return.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` to be used by the head.
      labels: Labels `Tensor`, or `dict` of same.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns an op
        to optimize the model with the loss in TRAIN mode. Used if `optimizer`
        is `None`. Exactly one of `train_op_fn` and `optimizer` must be set in
        TRAIN mode. None is allowed in other modes. If you want to optimize loss
        yourself you can pass `lambda _: tf.no_op()` and then use
          EstimatorSpec.loss to compute and apply gradients.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses.

    Returns:
      A `model_fn._TPUEstimatorSpec' instance.
    """
        raise NotImplementedError('TPUEstimatorSpec not available for this model head.')