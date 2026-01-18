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
class RecallAtPrecision(SensitivitySpecificityBase):
    """Computes best recall where precision is >= specified value.

  For a given score-label-distribution the required precision might not
  be achievable, in this case 0.0 is returned as recall.

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the
  recall at the given precision. The threshold for the given precision
  value is computed and used to evaluate the corresponding recall.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  If `class_id` is specified, we calculate precision by considering only the
  entries in the batch for which `class_id` is above the threshold predictions,
  and computing the fraction of them for which `class_id` is indeed a correct
  label.

  Args:
    precision: A scalar value in range `[0, 1]`.
    num_thresholds: (Optional) Defaults to 200. The number of thresholds to
      use for matching the given precision.
    class_id: (Optional) Integer class ID for which we want binary metrics.
      This must be in the half-open interval `[0, num_classes)`, where
      `num_classes` is the last dimension of predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.RecallAtPrecision(0.8)
  >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
  >>> m.result().numpy()
  0.5

  >>> m.reset_state()
  >>> m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9],
  ...                sample_weight=[1, 0, 0, 1])
  >>> m.result().numpy()
  1.0

  Usage with `compile()` API:

  ```python
  model.compile(
      optimizer='sgd',
      loss='mse',
      metrics=[tf.keras.metrics.RecallAtPrecision(precision=0.8)])
  ```
  """

    def __init__(self, precision, num_thresholds=200, class_id=None, name=None, dtype=None):
        if precision < 0 or precision > 1:
            raise ValueError('`precision` must be in the range [0, 1].')
        self.precision = precision
        self.num_thresholds = num_thresholds
        super(RecallAtPrecision, self).__init__(value=precision, num_thresholds=num_thresholds, class_id=class_id, name=name, dtype=dtype)

    def result(self):
        precisions = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_positives)
        recalls = math_ops.div_no_nan(self.true_positives, self.true_positives + self.false_negatives)
        return self._find_max_under_constraint(precisions, recalls, math_ops.greater_equal)

    def get_config(self):
        config = {'num_thresholds': self.num_thresholds, 'precision': self.precision}
        base_config = super(RecallAtPrecision, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))