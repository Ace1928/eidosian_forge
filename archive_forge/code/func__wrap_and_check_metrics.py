import abc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.saving.utils_v1 import signature_def_utils as unexported_signature_utils
from tensorflow.python.saved_model import signature_def_utils
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