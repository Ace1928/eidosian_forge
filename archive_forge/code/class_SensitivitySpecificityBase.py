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
class SensitivitySpecificityBase(Metric, metaclass=abc.ABCMeta):
    """Abstract base class for computing sensitivity and specificity.

  For additional information about specificity and sensitivity, see
  [the following](https://en.wikipedia.org/wiki/Sensitivity_and_specificity).
  """

    def __init__(self, value, num_thresholds=200, class_id=None, name=None, dtype=None):
        super(SensitivitySpecificityBase, self).__init__(name=name, dtype=dtype)
        if num_thresholds <= 0:
            raise ValueError('`num_thresholds` must be > 0.')
        self.value = value
        self.class_id = class_id
        self.true_positives = self.add_weight('true_positives', shape=(num_thresholds,), initializer=init_ops.zeros_initializer)
        self.true_negatives = self.add_weight('true_negatives', shape=(num_thresholds,), initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight('false_positives', shape=(num_thresholds,), initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight('false_negatives', shape=(num_thresholds,), initializer=init_ops.zeros_initializer)
        if num_thresholds == 1:
            self.thresholds = [0.5]
            self._thresholds_distributed_evenly = False
        else:
            thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
            self.thresholds = [0.0] + thresholds + [1.0]
            self._thresholds_distributed_evenly = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
        return metrics_utils.update_confusion_matrix_variables({metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives, metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives, metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives, metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives}, y_true, y_pred, thresholds=self.thresholds, thresholds_distributed_evenly=self._thresholds_distributed_evenly, class_id=self.class_id, sample_weight=sample_weight)

    def reset_state(self):
        num_thresholds = len(self.thresholds)
        confusion_matrix_variables = (self.true_positives, self.true_negatives, self.false_positives, self.false_negatives)
        backend.batch_set_value([(v, np.zeros((num_thresholds,))) for v in confusion_matrix_variables])

    def get_config(self):
        config = {'class_id': self.class_id}
        base_config = super(SensitivitySpecificityBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _find_max_under_constraint(self, constrained, dependent, predicate):
        """Returns the maximum of dependent_statistic that satisfies the constraint.

    Args:
      constrained: Over these values the constraint
        is specified. A rank-1 tensor.
      dependent: From these values the maximum that satiesfies the
        constraint is selected. Values in this tensor and in
        `constrained` are linked by having the same threshold at each
        position, hence this tensor must have the same shape.
      predicate: A binary boolean functor to be applied to arguments
      `constrained` and `self.value`, e.g. `tf.greater`.

    Returns maximal dependent value, if no value satiesfies the constraint 0.0.
    """
        feasible = array_ops.where_v2(predicate(constrained, self.value))
        feasible_exists = math_ops.greater(array_ops.size(feasible), 0)
        max_dependent = math_ops.reduce_max(array_ops.gather(dependent, feasible))
        return array_ops.where_v2(feasible_exists, max_dependent, 0.0)