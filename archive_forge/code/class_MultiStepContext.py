import functools
import sys
import time
import six
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc
class MultiStepContext(object):
    """A context object that can be used to capture things when running steps.

  This context object is useful when running multiple steps at a time using the
  `experimental_run_steps_on_iterator` API. For e.g. it allows the user's step
  function to specify which outputs to emit at what frequency. Currently it
  supports capturing output from the last step, as well as capturing non tensor
  outputs.  In the future it will be augmented to support other use cases such
  as output each N steps.
  """

    def __init__(self):
        """Initialize an output context.

    Returns:
      A context object.
    """
        self._last_step_outputs = {}
        self._last_step_outputs_reduce_ops = {}
        self._non_tensor_outputs = {}

    @property
    def last_step_outputs(self):
        """A dictionary consisting of outputs to be captured on last step.

    Keys in the dictionary are names of tensors to be captured, as specified
    when `set_last_step_output` is called.
    Values in the dictionary are the tensors themselves. If
    `set_last_step_output` was called with a `reduce_op` for this output,
    then the value is the reduced value.

    Returns:
      A dictionary with last step outputs.
    """
        return self._last_step_outputs

    def _set_last_step_outputs(self, outputs):
        """Replace the entire dictionary of last step outputs."""
        if not isinstance(outputs, dict):
            raise ValueError('Need a dictionary to set last_step_outputs.')
        self._last_step_outputs = outputs

    def set_last_step_output(self, name, output, reduce_op=None):
        """Set `output` with `name` to be outputted from the last step.

    Args:
      name: String, name to identify the output. Doesn't need to match tensor
        name.
      output: The tensors that should be outputted with `name`. See below for
        actual types supported.
      reduce_op: Reduction method to use to reduce outputs from multiple
        replicas. Required if `set_last_step_output` is called in a replica
        context. Optional in cross_replica_context.
        When present, the outputs from all the replicas are reduced using the
        current distribution strategy's `reduce` method. Hence, the type of
        `output` must be what's supported by the corresponding `reduce` method.
        For e.g. if using MirroredStrategy and reduction is set, output
        must be a `PerReplica` value.
        The reduce method is also recorded in a dictionary
        `_last_step_outputs_reduce_ops` for later interpreting of the
        outputs as already reduced or not.
    """
        if distribute_lib.in_cross_replica_context():
            self._last_step_outputs_reduce_ops[name] = reduce_op
            if reduce_op is None:
                self._last_step_outputs[name] = output
            else:
                distribution = distribute_lib.get_strategy()
                self._last_step_outputs[name] = distribution.reduce(reduce_op, output, axis=None)
        else:
            assert reduce_op is not None

            def merge_fn(distribution, value):
                self._last_step_outputs[name] = distribution.reduce(reduce_op, value, axis=None)
                self._last_step_outputs_reduce_ops[name] = reduce_op
            distribute_lib.get_replica_context().merge_call(merge_fn, args=(output,))

    @property
    def non_tensor_outputs(self):
        """A dictionary consisting of any non tensor outputs to be captured."""
        return self._non_tensor_outputs

    def set_non_tensor_output(self, name, output):
        """Set `output` with `name` to be captured as a non tensor output."""
        if distribute_lib.in_cross_replica_context():
            self._non_tensor_outputs[name] = output
        else:

            def merge_fn(distribution, value):
                self._non_tensor_outputs[name] = distribution.experimental_local_results(value)
            distribute_lib.get_replica_context().merge_call(merge_fn, args=(output,))