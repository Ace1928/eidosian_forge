import functools
import hashlib
import threading
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['ConditionalAccumulator'])
class ConditionalAccumulator(ConditionalAccumulatorBase):
    """A conditional accumulator for aggregating gradients.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
  """

    def __init__(self, dtype, shape=None, shared_name=None, name='conditional_accumulator', reduction_type='MEAN'):
        """Creates a new ConditionalAccumulator.

    Args:
      dtype: Datatype of the accumulated gradients.
      shape: Shape of the accumulated gradients.
      shared_name: Optional. If non-empty, this accumulator will be shared under
        the given name across multiple sessions.
      name: Optional name for the accumulator.
      reduction_type: Reduction type to use when taking the gradient.
    """
        accumulator_ref = gen_data_flow_ops.resource_conditional_accumulator(dtype=dtype, shape=shape, shared_name=shared_name, name=name, reduction_type=reduction_type)
        if context.executing_eagerly():
            self._resource_deleter = resource_variable_ops.EagerResourceDeleter(handle=accumulator_ref, handle_device=context.context().device_name)
        super(ConditionalAccumulator, self).__init__(dtype, shape, accumulator_ref)

    def apply_grad(self, grad, local_step=0, name=None):
        """Attempts to apply a gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., local_step
    is less than the accumulator's global time step.

    Args:
      grad: The gradient tensor to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      ValueError: If grad is of the wrong shape
    """
        grad = ops.convert_to_tensor(grad, self._dtype)
        grad.get_shape().assert_is_compatible_with(self._shape)
        local_step = math_ops.cast(ops.convert_to_tensor(local_step), _dtypes.int64)
        return gen_data_flow_ops.resource_accumulator_apply_gradient(self._accumulator_ref, local_step=local_step, gradient=grad, name=name)

    def take_grad(self, num_required, name=None):
        """Attempts to extract the average gradient from the accumulator.

    The operation blocks until sufficient number of gradients have been
    successfully applied to the accumulator.

    Once successful, the following actions are also triggered:

    - Counter of accumulated gradients is reset to 0.
    - Aggregated gradient is reset to 0 tensor.
    - Accumulator's internal time step is incremented by 1.

    Args:
      num_required: Number of gradients that needs to have been aggregated
      name: Optional name for the operation

    Returns:
      A tensor holding the value of the average gradient.

    Raises:
      InvalidArgumentError: If num_required < 1
    """
        out = gen_data_flow_ops.resource_accumulator_take_gradient(self._accumulator_ref, num_required, dtype=self._dtype, name=name)
        out.set_shape(self._shape)
        return out