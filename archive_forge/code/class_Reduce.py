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
class Reduce(Metric):
    """Encapsulates metrics that perform a reduce operation on the values.

  Args:
    reduction: a `tf.keras.metrics.Reduction` enum value.
    name: string name of the metric instance.
    dtype: (Optional) data type of the metric result.
  """

    def __init__(self, reduction, name, dtype=None):
        super(Reduce, self).__init__(name=name, dtype=dtype)
        self.reduction = reduction
        self.total = self.add_weight('total', initializer=init_ops.zeros_initializer)
        if reduction in [metrics_utils.Reduction.SUM_OVER_BATCH_SIZE, metrics_utils.Reduction.WEIGHTED_MEAN]:
            self.count = self.add_weight('count', initializer=init_ops.zeros_initializer)

    def update_state(self, values, sample_weight=None):
        """Accumulates statistics for computing the metric.

    Args:
      values: Per-example value.
      sample_weight: Optional weighting of each example. Defaults to 1.

    Returns:
      Update op.
    """
        [values], sample_weight = metrics_utils.ragged_assert_compatible_and_get_flat_values([values], sample_weight)
        try:
            values = math_ops.cast(values, self._dtype)
        except (ValueError, TypeError):
            msg = 'The output of a metric function can only be a single Tensor. Got: %s' % (values,)
            if isinstance(values, dict):
                msg += '. To return a dict of values, implement a custom Metric subclass.'
            raise RuntimeError(msg)
        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            values, _, sample_weight = losses_utils.squeeze_or_expand_dimensions(values, sample_weight=sample_weight)
            try:
                sample_weight = weights_broadcast_ops.broadcast_weights(sample_weight, values)
            except ValueError:
                ndim = backend.ndim(values)
                weight_ndim = backend.ndim(sample_weight)
                if self.reduction == metrics_utils.Reduction.SUM:
                    values = math_ops.reduce_sum(values, axis=list(range(weight_ndim, ndim)))
                else:
                    values = math_ops.reduce_mean(values, axis=list(range(weight_ndim, ndim)))
            values = math_ops.multiply(values, sample_weight)
        value_sum = math_ops.reduce_sum(values)
        with ops.control_dependencies([value_sum]):
            update_total_op = self.total.assign_add(value_sum)
        if self.reduction == metrics_utils.Reduction.SUM:
            return update_total_op
        if self.reduction == metrics_utils.Reduction.SUM_OVER_BATCH_SIZE:
            num_values = math_ops.cast(array_ops.size(values), self._dtype)
        elif self.reduction == metrics_utils.Reduction.WEIGHTED_MEAN:
            if sample_weight is None:
                num_values = math_ops.cast(array_ops.size(values), self._dtype)
            else:
                num_values = math_ops.reduce_sum(sample_weight)
        else:
            raise NotImplementedError('reduction [%s] not implemented' % self.reduction)
        with ops.control_dependencies([update_total_op]):
            return self.count.assign_add(num_values)

    def result(self):
        if self.reduction == metrics_utils.Reduction.SUM:
            return array_ops.identity(self.total)
        elif self.reduction in [metrics_utils.Reduction.WEIGHTED_MEAN, metrics_utils.Reduction.SUM_OVER_BATCH_SIZE]:
            return math_ops.div_no_nan(self.total, self.count)
        else:
            raise NotImplementedError('reduction [%s] not implemented' % self.reduction)