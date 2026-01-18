import abc
import types
import warnings
import numpy as np
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
from tensorflow.python.keras.losses import hinge
from tensorflow.python.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras.losses import logcosh
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras.losses import poisson
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.losses import squared_hinge
from tensorflow.python.keras.saving.saved_model import metric_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
class _ConfusionMatrixConditionCount(Metric):
    """Calculates the number of the given confusion matrix condition.

  Args:
    confusion_matrix_cond: One of `metrics_utils.ConfusionMatrix` conditions.
    thresholds: (Optional) Defaults to 0.5. A float value or a python list/tuple
      of float threshold values in [0, 1]. A threshold is compared with
      prediction values to determine the truth value of predictions (i.e., above
      the threshold is `true`, below is `false`). One metric value is generated
      for each threshold value.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

    def __init__(self, confusion_matrix_cond, thresholds=None, name=None, dtype=None):
        super(_ConfusionMatrixConditionCount, self).__init__(name=name, dtype=dtype)
        self._confusion_matrix_cond = confusion_matrix_cond
        self.init_thresholds = thresholds
        self.thresholds = metrics_utils.parse_init_thresholds(thresholds, default_threshold=0.5)
        self._thresholds_distributed_evenly = metrics_utils.is_evenly_distributed_thresholds(self.thresholds)
        self.accumulator = self.add_weight('accumulator', shape=(len(self.thresholds),), initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the metric statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
        return metrics_utils.update_confusion_matrix_variables({self._confusion_matrix_cond: self.accumulator}, y_true, y_pred, thresholds=self.thresholds, thresholds_distributed_evenly=self._thresholds_distributed_evenly, sample_weight=sample_weight)

    def result(self):
        if len(self.thresholds) == 1:
            result = self.accumulator[0]
        else:
            result = self.accumulator
        return tensor_conversion.convert_to_tensor_v2_with_dispatch(result)

    def reset_state(self):
        num_thresholds = len(to_list(self.thresholds))
        backend.batch_set_value([(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {'thresholds': self.init_thresholds}
        base_config = super(_ConfusionMatrixConditionCount, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))