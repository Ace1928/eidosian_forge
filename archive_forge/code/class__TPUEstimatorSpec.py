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
class _TPUEstimatorSpec(collections.namedtuple('TPUEstimatorSpec', ['mode', 'predictions', 'loss', 'train_op', 'eval_metrics', 'export_outputs', 'scaffold_fn', 'host_call', 'training_hooks', 'evaluation_hooks', 'prediction_hooks'])):
    """Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.

  This is a simplified implementation of `tf.contrib.tpu.EstimatorSpec`. See
  tensorflow/contrib/tpu/python/tpu/tpu_estimator.py for more detailed
  documentation.
  """

    def __new__(cls, mode, predictions=None, loss=None, train_op=None, eval_metrics=None, export_outputs=None, scaffold_fn=None, host_call=None, training_hooks=None, evaluation_hooks=None, prediction_hooks=None):
        """Creates a `_TPUEstimatorSpec` instance."""
        train_op = _validate_estimator_spec_train_op(train_op, mode)
        loss = _validate_estimator_spec_loss(loss, mode)
        predictions = _validate_estimator_spec_predictions(predictions, mode)
        export_outputs = _validate_estimator_spec_export_outputs(export_outputs, predictions, mode)
        training_hooks = _validate_estimator_spec_hooks(training_hooks)
        evaluation_hooks = _validate_estimator_spec_hooks(evaluation_hooks)
        prediction_hooks = _validate_estimator_spec_hooks(prediction_hooks)
        return super(_TPUEstimatorSpec, cls).__new__(cls, mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metrics=eval_metrics, export_outputs=export_outputs, scaffold_fn=scaffold_fn, host_call=host_call, training_hooks=training_hooks, evaluation_hooks=evaluation_hooks, prediction_hooks=prediction_hooks)

    def as_estimator_spec(self):
        """Creates an equivalent `EstimatorSpec` used by CPU train/eval."""
        if not self.eval_metrics:
            eval_metric_ops = None
        else:
            metric_fn, tensors = self.eval_metrics
            eval_metric_ops = metric_fn(**tensors)
        return EstimatorSpec(mode=self.mode, predictions=self.predictions, loss=self.loss, train_op=self.train_op, eval_metric_ops=eval_metric_ops, export_outputs=self.export_outputs, training_hooks=self.training_hooks, evaluation_hooks=self.evaluation_hooks, prediction_hooks=self.prediction_hooks)