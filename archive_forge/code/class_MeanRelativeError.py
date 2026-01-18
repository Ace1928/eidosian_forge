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
class MeanRelativeError(Mean):
    """Computes the mean relative error by normalizing with the given values.

  This metric creates two local variables, `total` and `count` that are used to
  compute the mean relative error. This is weighted by `sample_weight`, and
  it is ultimately returned as `mean_relative_error`:
  an idempotent operation that simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Args:
    normalizer: The normalizer values with same shape as predictions.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.

  Standalone usage:

  >>> m = tf.keras.metrics.MeanRelativeError(normalizer=[1, 3, 2, 3])
  >>> m.update_state([1, 3, 2, 3], [2, 4, 6, 8])

  >>> # metric = mean(|y_pred - y_true| / normalizer)
  >>> #        = mean([1, 1, 4, 5] / [1, 3, 2, 3]) = mean([1, 1/3, 2, 5/3])
  >>> #        = 5/4 = 1.25
  >>> m.result().numpy()
  1.25

  Usage with `compile()` API:

  ```python
  model.compile(
    optimizer='sgd',
    loss='mse',
    metrics=[tf.keras.metrics.MeanRelativeError(normalizer=[1, 3])])
  ```
  """

    def __init__(self, normalizer, name=None, dtype=None):
        super(MeanRelativeError, self).__init__(name=name, dtype=dtype)
        normalizer = math_ops.cast(normalizer, self._dtype)
        self.normalizer = normalizer

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        [y_pred, y_true], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values([y_pred, y_true], sample_weight)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        y_pred, self.normalizer = losses_utils.remove_squeezable_dimensions(y_pred, self.normalizer)
        y_pred.shape.assert_is_compatible_with(y_true.shape)
        relative_errors = math_ops.div_no_nan(math_ops.abs(y_true - y_pred), self.normalizer)
        return super(MeanRelativeError, self).update_state(relative_errors, sample_weight=sample_weight)

    def get_config(self):
        n = self.normalizer
        config = {'normalizer': backend.eval(n) if is_tensor_or_variable(n) else n}
        base_config = super(MeanRelativeError, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))