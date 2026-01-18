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
class MeanMetricWrapper(Mean):
    """Wraps a stateless metric function with the Mean metric.

  You could use this class to quickly build a mean metric from a function. The
  function needs to have the signature `fn(y_true, y_pred)` and return a
  per-sample loss array. `MeanMetricWrapper.result()` will return
  the average metric value across all samples seen so far.

  For example:

  ```python
  def accuracy(y_true, y_pred):
    return tf.cast(tf.math.equal(y_true, y_pred), tf.float32)

  accuracy_metric = tf.keras.metrics.MeanMetricWrapper(fn=accuracy)

  keras_model.compile(..., metrics=accuracy_metric)
  ```

  Args:
    fn: The metric function to wrap, with signature `fn(y_true, y_pred,
      **kwargs)`.
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    **kwargs: Keyword arguments to pass on to `fn`.
  """

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
      sample_weight: Optional `sample_weight` acts as a
        coefficient for the metric. If a scalar is provided, then the metric is
        simply scaled by the given value. If `sample_weight` is a tensor of size
        `[batch_size]`, then the metric for each sample of the batch is rescaled
        by the corresponding element in the `sample_weight` vector. If the shape
        of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted
        to this shape), then each metric element of `y_pred` is scaled by the
        corresponding value of `sample_weight`. (Note on `dN-1`: all metric
        functions reduce by 1 dimension, usually the last axis (-1)).

    Returns:
      Update op.
    """
        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)
        [y_true, y_pred], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values([y_true, y_pred], sample_weight)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        ag_fn = autograph.tf_convert(self._fn, ag_ctx.control_status_ctx())
        matches = ag_fn(y_true, y_pred, **self._fn_kwargs)
        return super(MeanMetricWrapper, self).update_state(matches, sample_weight=sample_weight)

    def get_config(self):
        config = {}
        if type(self) is MeanMetricWrapper:
            config['fn'] = self._fn
        for k, v in self._fn_kwargs.items():
            config[k] = backend.eval(v) if is_tensor_or_variable(v) else v
        base_config = super(MeanMetricWrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        fn = config.pop('fn', None)
        if cls is MeanMetricWrapper:
            return cls(get(fn), **config)
        return super(MeanMetricWrapper, cls).from_config(config)