import abc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.saving.utils_v1 import signature_def_utils as unexported_signature_utils
from tensorflow.python.saved_model import signature_def_utils
class _SupervisedOutput(ExportOutput):
    """Represents the output of a supervised training or eval process."""
    __metaclass__ = abc.ABCMeta
    LOSS_NAME = 'loss'
    PREDICTIONS_NAME = 'predictions'
    METRICS_NAME = 'metrics'
    METRIC_VALUE_SUFFIX = 'value'
    METRIC_UPDATE_SUFFIX = 'update_op'
    _loss = None
    _predictions = None
    _metrics = None

    def __init__(self, loss=None, predictions=None, metrics=None):
        """Constructor for SupervisedOutput (ie, Train or Eval output).

    Args:
      loss: dict of Tensors or single Tensor representing calculated loss.
      predictions: dict of Tensors or single Tensor representing model
        predictions.
      metrics: Dict of metric results keyed by name.
        The values of the dict can be one of the following:
        (1) instance of `Metric` class.
        (2) (metric_value, update_op) tuples, or a single tuple.
        metric_value must be a Tensor, and update_op must be a Tensor or Op.

    Raises:
      ValueError: if any of the outputs' dict keys are not strings or tuples of
        strings or the values are not Tensors (or Operations in the case of
        update_op).
    """
        if loss is not None:
            loss_dict = self._wrap_and_check_outputs(loss, self.LOSS_NAME)
            self._loss = self._prefix_output_keys(loss_dict, self.LOSS_NAME)
        if predictions is not None:
            pred_dict = self._wrap_and_check_outputs(predictions, self.PREDICTIONS_NAME)
            self._predictions = self._prefix_output_keys(pred_dict, self.PREDICTIONS_NAME)
        if metrics is not None:
            self._metrics = self._wrap_and_check_metrics(metrics)

    def _prefix_output_keys(self, output_dict, output_name):
        """Prepend output_name to the output_dict keys if it doesn't exist.

    This produces predictable prefixes for the pre-determined outputs
    of SupervisedOutput.

    Args:
      output_dict: dict of string to Tensor, assumed valid.
      output_name: prefix string to prepend to existing keys.

    Returns:
      dict with updated keys and existing values.
    """
        new_outputs = {}
        for key, val in output_dict.items():
            key = self._prefix_key(key, output_name)
            new_outputs[key] = val
        return new_outputs

    def _prefix_key(self, key, output_name):
        if key.find(output_name) != 0:
            key = output_name + self._SEPARATOR_CHAR + key
        return key

    def _wrap_and_check_metrics(self, metrics):
        """Handle the saving of metrics.

    Metrics is either a tuple of (value, update_op), or a dict of such tuples.
    Here, we separate out the tuples and create a dict with names to tensors.

    Args:
      metrics: Dict of metric results keyed by name.
        The values of the dict can be one of the following:
        (1) instance of `Metric` class.
        (2) (metric_value, update_op) tuples, or a single tuple.
        metric_value must be a Tensor, and update_op must be a Tensor or Op.

    Returns:
      dict of output_names to tensors

    Raises:
      ValueError: if the dict key is not a string, or the metric values or ops
        are not tensors.
    """
        if not isinstance(metrics, dict):
            metrics = {self.METRICS_NAME: metrics}
        outputs = {}
        for key, value in metrics.items():
            if isinstance(value, tuple):
                metric_val, metric_op = value
            else:
                metric_val = value.result()
                assert len(value.updates) == 1
                metric_op = value.updates[0]
            key = self._check_output_key(key, self.METRICS_NAME)
            key = self._prefix_key(key, self.METRICS_NAME)
            val_name = key + self._SEPARATOR_CHAR + self.METRIC_VALUE_SUFFIX
            op_name = key + self._SEPARATOR_CHAR + self.METRIC_UPDATE_SUFFIX
            if not isinstance(metric_val, tensor.Tensor):
                raise ValueError('{} output value must be a Tensor; got {}.'.format(key, metric_val))
            if not (tensor_util.is_tensor(metric_op) or isinstance(metric_op, ops.Operation)):
                raise ValueError('{} update_op must be a Tensor or Operation; got {}.'.format(key, metric_op))
            metric_op_tensor = metric_op
            if not isinstance(metric_op, tensor.Tensor):
                with ops.control_dependencies([metric_op]):
                    metric_op_tensor = constant_op.constant([], name='metric_op_wrapper')
            outputs[val_name] = metric_val
            outputs[op_name] = metric_op_tensor
        return outputs

    @property
    def loss(self):
        return self._loss

    @property
    def predictions(self):
        return self._predictions

    @property
    def metrics(self):
        return self._metrics

    @abc.abstractmethod
    def _get_signature_def_fn(self):
        """Returns a function that produces a SignatureDef given desired outputs."""
        pass

    def as_signature_def(self, receiver_tensors):
        signature_def_fn = self._get_signature_def_fn()
        return signature_def_fn(receiver_tensors, self.loss, self.predictions, self.metrics)