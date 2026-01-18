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
class _RegressionHeadWithMeanSquaredErrorLoss(_Head):
    """`Head` for regression using the mean squared loss."""

    def __init__(self, label_dimension, weight_column=None, loss_reduction=tf.compat.v1.losses.Reduction.SUM, loss_fn=None, inverse_link_fn=None, name=None):
        """`Head` for regression."""
        if label_dimension < 1:
            raise ValueError('Invalid label_dimension %s.' % label_dimension)
        self._logits_dimension = label_dimension
        self._weight_column = weight_column
        self._loss_reduction = loss_reduction
        self._loss_fn = loss_fn
        self._inverse_link_fn = inverse_link_fn
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def logits_dimension(self):
        return self._logits_dimension

    def create_loss(self, features, mode, logits, labels):
        """See `Head`."""
        del mode
        logits = ops.convert_to_tensor(logits)
        labels = _check_dense_labels_match_logits_and_reshape(labels=labels, logits=logits, expected_labels_dimension=self._logits_dimension)
        labels = tf.cast(labels, dtype=tf.dtypes.float32)
        if self._loss_fn:
            unweighted_loss = _call_loss_fn(loss_fn=self._loss_fn, labels=labels, logits=logits, features=features, expected_loss_dim=self._logits_dimension)
        else:
            unweighted_loss = tf.compat.v1.losses.mean_squared_error(labels=labels, predictions=logits, reduction=tf.compat.v1.losses.Reduction.NONE)
        weights = _get_weights_and_check_match_logits(features=features, weight_column=self._weight_column, logits=logits, allow_per_logit_weights=True)
        training_loss = tf.compat.v1.losses.compute_weighted_loss(unweighted_loss, weights=weights, reduction=self._loss_reduction)
        return LossSpec(training_loss=training_loss, unreduced_loss=unweighted_loss, weights=weights, processed_labels=labels)

    def _eval_metric_ops(self, predicted_value, labels, weights, unreduced_loss, regularization_loss):
        """Returns the Eval metric ops."""
        keys = metric_keys.MetricKeys
        eval_metric_ops = {_summary_key(self._name, keys.LOSS_MEAN): tf.compat.v1.metrics.mean(values=unreduced_loss, weights=weights), _summary_key(self._name, keys.PREDICTION_MEAN): _predictions_mean(predictions=predicted_value, weights=weights, name=keys.PREDICTION_MEAN), _summary_key(self._name, keys.LABEL_MEAN): tf.compat.v1.metrics.mean(values=labels, weights=weights)}
        if regularization_loss is not None:
            regularization_loss_key = _summary_key(self._name, keys.LOSS_REGULARIZATION)
            eval_metric_ops[regularization_loss_key] = tf.compat.v1.metrics.mean(values=regularization_loss, name=keys.LOSS_REGULARIZATION)
        return eval_metric_ops

    def _create_tpu_estimator_spec(self, features, mode, logits, labels=None, optimizer=None, train_op_fn=None, regularization_losses=None):
        """Returns an `EstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.
      labels: Labels `Tensor` with shape matching `logits`, namely `[D0, D1, ...
        DN, logits_dimension]`. When `logits_dimension=1`, shape `[D0, D1, ...
        DN]` is also supported. `labels` is required argument when `mode` equals
        `TRAIN` or `EVAL`.
      optimizer: `Optimizer` instance to optimize the loss in TRAIN mode.
        Namely, sets `train_op = optimizer.minimize(loss, global_step)`, which
        updates variables and increments `global_step`.
      train_op_fn: Function that takes a scalar loss `Tensor` and returns
        `train_op`. Used if `optimizer` is `None`.
      regularization_losses: A list of additional scalar losses to be added to
        the training loss, such as regularization losses. These losses are
        usually expressed as a batch average, so for best results users need to
        set `loss_reduction=SUM_OVER_BATCH_SIZE` when creating the head to avoid
        scaling errors.

    Returns:
      A `model_fn._TPUEstimatorSpec` instance.
    Raises:
      ValueError: If both `train_op_fn` and `optimizer` are `None` in TRAIN
        mode, or if both are set.
    """
        with ops.name_scope(self._name, 'head'):
            logits = _check_logits_final_dim(logits, self._logits_dimension)
            if self._inverse_link_fn:
                predicted_value = self._inverse_link_fn(logits)
                predictions = {prediction_keys.PredictionKeys.PREDICTIONS: predicted_value, prediction_keys.PredictionKeys.LOGITS: logits}
            else:
                predicted_value = logits
                predictions = {prediction_keys.PredictionKeys.PREDICTIONS: predicted_value}
            if mode == ModeKeys.PREDICT:
                regression_output = export_output.RegressionOutput(value=predicted_value)
                return model_fn._TPUEstimatorSpec(mode=ModeKeys.PREDICT, predictions=predictions, export_outputs={_DEFAULT_SERVING_KEY: regression_output, _REGRESS_SERVING_KEY: regression_output, _PREDICT_SERVING_KEY: export_output.PredictOutput(predictions)})
            training_loss, unreduced_loss, weights, _ = self.create_loss(features=features, mode=mode, logits=logits, labels=labels)
            if regularization_losses:
                regularization_loss = tf.math.add_n(regularization_losses)
                regularized_training_loss = tf.math.add_n([training_loss, regularization_loss])
            else:
                regularization_loss = None
                regularized_training_loss = training_loss
            if mode == ModeKeys.EVAL:
                return model_fn._TPUEstimatorSpec(mode=ModeKeys.EVAL, predictions=predictions, loss=regularized_training_loss, eval_metrics=_create_eval_metrics_tuple(self._eval_metric_ops, {'predicted_value': predicted_value, 'labels': labels, 'weights': weights, 'unreduced_loss': unreduced_loss, 'regularization_loss': regularization_loss}))
            if optimizer is not None:
                if train_op_fn is not None:
                    raise ValueError('train_op_fn and optimizer cannot both be set.')
                train_op = optimizer.minimize(regularized_training_loss, global_step=tf.compat.v1.train.get_global_step())
            elif train_op_fn is not None:
                train_op = train_op_fn(regularized_training_loss)
            else:
                raise ValueError('train_op_fn and optimizer cannot both be None.')
            train_op = _append_update_ops(train_op)
            if self._loss_reduction == tf.compat.v1.losses.Reduction.SUM:
                example_weight_sum = tf.math.reduce_sum(weights * tf.compat.v1.ones_like(unreduced_loss))
                mean_loss = training_loss / example_weight_sum
            else:
                mean_loss = None
        with ops.name_scope(''):
            keys = metric_keys.MetricKeys
            tf.compat.v1.summary.scalar(_summary_key(self._name, keys.LOSS), regularized_training_loss)
            if mean_loss is not None:
                tf.compat.v1.summary.scalar(_summary_key(self._name, keys.LOSS_MEAN), mean_loss)
            if regularization_loss is not None:
                tf.compat.v1.summary.scalar(_summary_key(self._name, keys.LOSS_REGULARIZATION), regularization_loss)
        return model_fn._TPUEstimatorSpec(mode=ModeKeys.TRAIN, predictions=predictions, loss=regularized_training_loss, train_op=train_op)