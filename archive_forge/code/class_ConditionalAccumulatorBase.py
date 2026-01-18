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
@tf_export(v1=['ConditionalAccumulatorBase'])
class ConditionalAccumulatorBase:
    """A conditional accumulator for aggregating gradients.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
  """

    def __init__(self, dtype, shape, accumulator_ref):
        """Creates a new ConditionalAccumulator.

    Args:
      dtype: Datatype of the accumulated gradients.
      shape: Shape of the accumulated gradients.
      accumulator_ref: A handle to the conditional accumulator, created by sub-
        classes
    """
        self._dtype = dtype
        if shape is not None:
            self._shape = tensor_shape.TensorShape(shape)
        else:
            self._shape = tensor_shape.unknown_shape()
        self._accumulator_ref = accumulator_ref
        if context.executing_eagerly():
            self._name = context.context().scope_name
        else:
            self._name = self._accumulator_ref.op.name.split('/')[-1]

    @property
    def accumulator_ref(self):
        """The underlying accumulator reference."""
        return self._accumulator_ref

    @property
    def name(self):
        """The name of the underlying accumulator."""
        return self._name

    @property
    def dtype(self):
        """The datatype of the gradients accumulated by this accumulator."""
        return self._dtype

    def num_accumulated(self, name=None):
        """Number of gradients that have currently been aggregated in accumulator.

    Args:
      name: Optional name for the operation.

    Returns:
      Number of accumulated gradients currently in accumulator.
    """
        if name is None:
            name = '%s_NumAccumulated' % self._name
        return gen_data_flow_ops.resource_accumulator_num_accumulated(self._accumulator_ref, name=name)

    def set_global_step(self, new_global_step, name=None):
        """Sets the global time step of the accumulator.

    The operation logs a warning if we attempt to set to a time step that is
    lower than the accumulator's own time step.

    Args:
      new_global_step: Value of new time step. Can be a variable or a constant
      name: Optional name for the operation.

    Returns:
      Operation that sets the accumulator's time step.
    """
        return gen_data_flow_ops.resource_accumulator_set_global_step(self._accumulator_ref, math_ops.cast(ops.convert_to_tensor(new_global_step), _dtypes.int64), name=name)