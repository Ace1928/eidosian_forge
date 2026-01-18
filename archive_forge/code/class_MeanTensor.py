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
class MeanTensor(Metric):
    """Computes the element-wise (weighted) mean of the given tensors.

  `MeanTensor` returns a tensor with the same shape of the input tensors. The
  mean value is updated by keeping local variables `total` and `count`. The
  `total` tracks the sum of the weighted values, and `count` stores the sum of
  the weighted counts.

  Args:
    name: (Optional) string name of the metric instance.
    dtype: (Optional) data type of the metric result.
    shape: (Optional) A list of integers, a tuple of integers, or a 1-D Tensor
      of type int32. If not specified, the shape is inferred from the values at
      the first call of update_state.

  Standalone usage:

  >>> m = tf.keras.metrics.MeanTensor()
  >>> m.update_state([0, 1, 2, 3])
  >>> m.update_state([4, 5, 6, 7])
  >>> m.result().numpy()
  array([2., 3., 4., 5.], dtype=float32)

  >>> m.update_state([12, 10, 8, 6], sample_weight= [0, 0.2, 0.5, 1])
  >>> m.result().numpy()
  array([2.       , 3.6363635, 4.8      , 5.3333335], dtype=float32)

  >>> m = tf.keras.metrics.MeanTensor(dtype=tf.float64, shape=(1, 4))
  >>> m.result().numpy()
  array([[0., 0., 0., 0.]])
  >>> m.update_state([[0, 1, 2, 3]])
  >>> m.update_state([[4, 5, 6, 7]])
  >>> m.result().numpy()
  array([[2., 3., 4., 5.]])
  """

    def __init__(self, name='mean_tensor', dtype=None, shape=None):
        super(MeanTensor, self).__init__(name=name, dtype=dtype)
        self._shape = None
        self._total = None
        self._count = None
        self._built = False
        if shape is not None:
            self._build(shape)

    def _build(self, shape):
        self._shape = tensor_shape.TensorShape(shape)
        self._build_input_shape = self._shape
        self._total = self.add_weight('total', shape=shape, initializer=init_ops.zeros_initializer)
        self._count = self.add_weight('count', shape=shape, initializer=init_ops.zeros_initializer)
        with ops.init_scope():
            if not context.executing_eagerly():
                backend._initialize_variables(backend._get_session())
        self._built = True

    @property
    def total(self):
        return self._total if self._built else None

    @property
    def count(self):
        return self._count if self._built else None

    def update_state(self, values, sample_weight=None):
        """Accumulates statistics for computing the element-wise mean.

    Args:
      values: Per-example value.
      sample_weight: Optional weighting of each example. Defaults to 1.

    Returns:
      Update op.
    """
        values = math_ops.cast(values, self._dtype)
        if not self._built:
            self._build(values.shape)
        elif values.shape != self._shape:
            raise ValueError('MeanTensor input values must always have the same shape. Expected shape (set during the first call): {}. Got: {}'.format(self._shape, values.shape))
        num_values = array_ops.ones_like(values)
        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            values, _, sample_weight = losses_utils.squeeze_or_expand_dimensions(values, sample_weight=sample_weight)
            try:
                sample_weight = weights_broadcast_ops.broadcast_weights(sample_weight, values)
            except ValueError:
                ndim = backend.ndim(values)
                weight_ndim = backend.ndim(sample_weight)
                values = math_ops.reduce_mean(values, axis=list(range(weight_ndim, ndim)))
            num_values = math_ops.multiply(num_values, sample_weight)
            values = math_ops.multiply(values, sample_weight)
        update_total_op = self._total.assign_add(values)
        with ops.control_dependencies([update_total_op]):
            return self._count.assign_add(num_values)

    def result(self):
        if not self._built:
            raise ValueError('MeanTensor does not have any result yet. Please call the MeanTensor instance or use `.update_state(value)` before retrieving the result.')
        return math_ops.div_no_nan(self.total, self.count)

    def reset_state(self):
        if self._built:
            backend.batch_set_value([(v, np.zeros(self._shape.as_list())) for v in self.variables])