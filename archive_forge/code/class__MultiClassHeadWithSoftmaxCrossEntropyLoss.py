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
class _MultiClassHeadWithSoftmaxCrossEntropyLoss(_Head):
    """See `_multi_class_head_with_softmax_cross_entropy_loss`."""

    def __init__(self, n_classes, weight_column=None, label_vocabulary=None, loss_reduction=tf.compat.v1.losses.Reduction.SUM, loss_fn=None, name=None):
        if n_classes is None:
            raise ValueError('n_classes cannot be None')
        self._n_classes = _validate_n_classes(n_classes)
        self._weight_column = weight_column
        self._label_vocabulary = label_vocabulary
        self._loss_reduction = loss_reduction
        self._loss_fn = loss_fn
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def logits_dimension(self):
        return self._n_classes

    def _eval_metric_ops(self, labels, class_ids, weights, unreduced_loss, regularization_loss):
        """Returns the Eval metric ops."""
        with ops.name_scope(None, 'metrics', (labels, class_ids, weights, unreduced_loss, regularization_loss)):
            keys = metric_keys.MetricKeys
            metric_ops = {_summary_key(self._name, keys.LOSS_MEAN): tf.compat.v1.metrics.mean(values=unreduced_loss, weights=weights, name=keys.LOSS_MEAN), _summary_key(self._name, keys.ACCURACY): tf.compat.v1.metrics.accuracy(labels=labels, predictions=class_ids, weights=weights, name=keys.ACCURACY)}
            if regularization_loss is not None:
                metric_ops[_summary_key(self._name, keys.LOSS_REGULARIZATION)] = tf.compat.v1.metrics.mean(values=regularization_loss, name=keys.LOSS_REGULARIZATION)
        return metric_ops

    def _label_ids(self, labels):
        """Converts labels to integer id space."""
        if self._label_vocabulary is None:
            if not labels.dtype.is_integer:
                raise ValueError('Labels dtype should be integer. Instead got {}.'.format(labels.dtype))
            label_ids = labels
        else:
            if labels.dtype != tf.dtypes.string:
                raise ValueError('Labels dtype should be string if there is a vocabulary. Instead got {}'.format(labels.dtype))
            label_ids = lookup_ops.index_table_from_tensor(vocabulary_list=tuple(self._label_vocabulary), name='class_id_lookup').lookup(labels)
        return _assert_range(label_ids, self._n_classes)

    def create_loss(self, features, mode, logits, labels):
        """See `Head`."""
        del mode
        logits = ops.convert_to_tensor(logits)
        labels = _check_dense_labels_match_logits_and_reshape(labels=labels, logits=logits, expected_labels_dimension=1)
        label_ids = self._label_ids(labels)
        if self._loss_fn:
            unweighted_loss = _call_loss_fn(loss_fn=self._loss_fn, labels=label_ids, logits=logits, features=features, expected_loss_dim=1)
        else:
            unweighted_loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=label_ids, logits=logits, reduction=tf.compat.v1.losses.Reduction.NONE)
            unweighted_loss = tf.compat.v1.expand_dims(unweighted_loss, axis=-1)
        weights = _get_weights_and_check_match_logits(features=features, weight_column=self._weight_column, logits=logits)
        training_loss = tf.compat.v1.losses.compute_weighted_loss(unweighted_loss, weights=weights, reduction=self._loss_reduction)
        return LossSpec(training_loss=training_loss, unreduced_loss=unweighted_loss, weights=weights, processed_labels=label_ids)

    def _create_tpu_estimator_spec(self, features, mode, logits, labels=None, optimizer=None, train_op_fn=None, regularization_losses=None):
        """Returns a `model_fn._TPUEstimatorSpec`.

    Args:
      features: Input `dict` of `Tensor` or `SparseTensor` objects.
      mode: Estimator's `ModeKeys`.
      logits: logits `Tensor` with shape `[D0, D1, ... DN, logits_dimension]`.
        For many applications, the shape is `[batch_size, logits_dimension]`.
      labels: Labels integer or string `Tensor` with shape matching `logits`,
        namely `[D0, D1, ... DN, 1]` or `[D0, D1, ... DN]`. `labels` is required
        argument when `mode` equals `TRAIN` or `EVAL`.
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
            logits = _check_logits_final_dim(logits, self.logits_dimension)
            pred_keys = prediction_keys.PredictionKeys
            with ops.name_scope(None, 'predictions', (logits,)):
                all_class_ids = _all_class_ids(logits, self._n_classes)
                all_classes = _all_classes(logits, self._n_classes, label_vocabulary=self._label_vocabulary)
                class_ids = tf.compat.v1.math.argmax(logits, axis=-1, name=pred_keys.CLASS_IDS)
                class_ids = tf.compat.v1.expand_dims(class_ids, axis=-1)
                if self._label_vocabulary:
                    table = lookup_ops.index_to_string_table_from_tensor(vocabulary_list=self._label_vocabulary, name='class_string_lookup')
                    classes = table.lookup(class_ids)
                else:
                    classes = tf.strings.as_string(class_ids, name='str_classes')
                probabilities = tf.compat.v1.nn.softmax(logits, name=pred_keys.PROBABILITIES)
                predictions = {pred_keys.LOGITS: logits, pred_keys.PROBABILITIES: probabilities, pred_keys.CLASS_IDS: class_ids, pred_keys.CLASSES: classes, pred_keys.ALL_CLASS_IDS: all_class_ids, pred_keys.ALL_CLASSES: all_classes}
            if mode == ModeKeys.PREDICT:
                classifier_output = _classification_output(scores=probabilities, n_classes=self._n_classes, label_vocabulary=self._label_vocabulary)
                return model_fn._TPUEstimatorSpec(mode=ModeKeys.PREDICT, predictions=predictions, export_outputs={_DEFAULT_SERVING_KEY: classifier_output, _CLASSIFY_SERVING_KEY: classifier_output, _PREDICT_SERVING_KEY: export_output.PredictOutput(predictions)})
            training_loss, unreduced_loss, weights, label_ids = self.create_loss(features=features, mode=mode, logits=logits, labels=labels)
            if regularization_losses:
                regularization_loss = tf.math.add_n(regularization_losses)
                regularized_training_loss = tf.math.add_n([training_loss, regularization_loss])
            else:
                regularization_loss = None
                regularized_training_loss = training_loss
            if mode == ModeKeys.EVAL:
                return model_fn._TPUEstimatorSpec(mode=ModeKeys.EVAL, predictions=predictions, loss=regularized_training_loss, eval_metrics=_create_eval_metrics_tuple(self._eval_metric_ops, {'labels': label_ids, 'class_ids': class_ids, 'weights': weights, 'unreduced_loss': unreduced_loss, 'regularization_loss': regularization_loss}))
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