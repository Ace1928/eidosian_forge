import abc
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
@deprecation.deprecated_endpoints('mixed_precision.experimental.FixedLossScale', 'train.experimental.FixedLossScale')
@tf_export(v1=['mixed_precision.FixedLossScale', 'mixed_precision.experimental.FixedLossScale', 'train.experimental.FixedLossScale'])
class FixedLossScale(LossScale):
    """Loss scale with a fixed value.

  The loss scale is not updated for the lifetime of instances of this class.
  A given instance of this class always returns the same number when called.
  """

    @deprecation.deprecated(None, 'Use tf.keras.mixed_precision.LossScaleOptimizer instead. LossScaleOptimizer now has all the functionality of FixedLossScale')
    def __init__(self, loss_scale_value):
        """Creates the fixed loss scale.

    Args:
      loss_scale_value: A Python float. Its ideal value varies depending on
        models to run. Choosing a too small loss_scale might affect model
        quality; a too big loss_scale might cause inf or nan. There is no single
        right loss_scale to apply. There is no harm choosing a relatively big
        number as long as no nan or inf is encountered in training.

    Raises:
      ValueError: If loss_scale_value is less than 1.
    """
        super(FixedLossScale, self).__init__()
        if not isinstance(loss_scale_value, (int, float)):
            raise ValueError('loss_scale_value must be a Python int or float.')
        if loss_scale_value < 1:
            raise ValueError('loss_scale_value must be at least 1.')
        self._loss_scale_value = float(loss_scale_value)

    def __call__(self):
        return ops.convert_to_tensor(self._loss_scale_value)

    def update(self, grads):
        del grads
        return (control_flow_ops.no_op(), True)

    def __repr__(self):
        return 'FixedLossScale(%s)' % self._loss_scale_value

    def get_config(self):
        return {'loss_scale_value': self._loss_scale_value}