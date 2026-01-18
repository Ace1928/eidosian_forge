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
@estimator_export('estimator.EstimatorSpec')
class EstimatorSpec(collections.namedtuple('EstimatorSpec', ['mode', 'predictions', 'loss', 'train_op', 'eval_metric_ops', 'export_outputs', 'training_chief_hooks', 'training_hooks', 'scaffold', 'evaluation_hooks', 'prediction_hooks'])):
    """Ops and objects returned from a `model_fn` and passed to an `Estimator`.

  `EstimatorSpec` fully defines the model to be run by an `Estimator`.
  """

    def __new__(cls, mode, predictions=None, loss=None, train_op=None, eval_metric_ops=None, export_outputs=None, training_chief_hooks=None, training_hooks=None, scaffold=None, evaluation_hooks=None, prediction_hooks=None):
        """Creates a validated `EstimatorSpec` instance.

    Depending on the value of `mode`, different arguments are required. Namely

    * For `mode == ModeKeys.TRAIN`: required fields are `loss` and `train_op`.
    * For `mode == ModeKeys.EVAL`: required field is `loss`.
    * For `mode == ModeKeys.PREDICT`: required fields are `predictions`.

    model_fn can populate all arguments independent of mode. In this case, some
    arguments will be ignored by an `Estimator`. E.g. `train_op` will be
    ignored in eval and infer modes. Example:

    ```python
    def my_model_fn(features, labels, mode):
      predictions = ...
      loss = ...
      train_op = ...
      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)
    ```

    Alternatively, model_fn can just populate the arguments appropriate to the
    given mode. Example:

    ```python
    def my_model_fn(features, labels, mode):
      if (mode == tf.estimator.ModeKeys.TRAIN or
          mode == tf.estimator.ModeKeys.EVAL):
        loss = ...
      else:
        loss = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = ...
      else:
        train_op = None
      if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = ...
      else:
        predictions = None

      return tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=predictions,
          loss=loss,
          train_op=train_op)
    ```

    Args:
      mode: A `ModeKeys`. Specifies if this is training, evaluation or
        prediction.
      predictions: Predictions `Tensor` or dict of `Tensor`.
      loss: Training loss `Tensor`. Must be either scalar, or with shape `[1]`.
      train_op: Op for the training step.
      eval_metric_ops: Dict of metric results keyed by name.
        The values of the dict can be one of the following: (1) instance of
          `Metric` class. (2) Results of calling a metric function, namely a
          `(metric_tensor, update_op)` tuple. `metric_tensor` should be
          evaluated without any impact on state (typically is a pure computation
          results based on variables.). For example, it should not trigger the
          `update_op` or requires any input fetching.
      export_outputs: Describes the output signatures to be exported to
        `SavedModel` and used during serving.
        A dict `{name: output}` where:
        * name: An arbitrary name for this output.
        * output: an `ExportOutput` object such as `ClassificationOutput`,
          `RegressionOutput`, or `PredictOutput`. Single-headed models only need
          to specify one entry in this dictionary. Multi-headed models should
          specify one entry for each head, one of which must be named using
          `tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`.
          If no entry is provided, a default `PredictOutput` mapping to
          `predictions` will be created.
      training_chief_hooks: Iterable of `tf.train.SessionRunHook` objects to run
        on the chief worker during training.
      training_hooks: Iterable of `tf.train.SessionRunHook` objects to run on
        all workers during training.
      scaffold: A `tf.train.Scaffold` object that can be used to set
        initialization, saver, and more to be used in training.
      evaluation_hooks: Iterable of `tf.train.SessionRunHook` objects to run
        during evaluation.
      prediction_hooks: Iterable of `tf.train.SessionRunHook` objects to run
        during predictions.

    Returns:
      A validated `EstimatorSpec` object.

    Raises:
      ValueError: If validation fails.
      TypeError: If any of the arguments is not the expected type.
    """
        train_op = _validate_estimator_spec_train_op(train_op, mode)
        loss = _validate_estimator_spec_loss(loss, mode)
        predictions = _validate_estimator_spec_predictions(predictions, mode)
        export_outputs = _validate_estimator_spec_export_outputs(export_outputs, predictions, mode)
        training_hooks = _validate_estimator_spec_hooks(training_hooks)
        evaluation_hooks = _validate_estimator_spec_hooks(evaluation_hooks)
        prediction_hooks = _validate_estimator_spec_hooks(prediction_hooks)
        training_chief_hooks = _validate_estimator_spec_hooks(training_chief_hooks)
        eval_metric_ops = _validate_eval_metric_ops(eval_metric_ops)
        scaffold = _validate_scaffold(scaffold)
        if tensor_tracer.TensorTracer.is_enabled() and train_op is not None:
            tt = tensor_tracer.TensorTracer()
            loss = tt.trace_cpu(tf.compat.v1.get_default_graph(), loss, train_op)
        return super(EstimatorSpec, cls).__new__(cls, mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops, export_outputs=export_outputs, training_chief_hooks=training_chief_hooks, training_hooks=training_hooks, scaffold=scaffold, evaluation_hooks=evaluation_hooks, prediction_hooks=prediction_hooks)

    def _replace(self, **kwds):
        """Return a new EstimatorSpec replacing specified fields with new values."""
        if 'mode' in kwds:
            if self.mode != kwds['mode']:
                raise ValueError('mode of EstimatorSpec cannot be changed.')
        new_fields = map(kwds.pop, self._fields, list(self))
        return EstimatorSpec(*new_fields)