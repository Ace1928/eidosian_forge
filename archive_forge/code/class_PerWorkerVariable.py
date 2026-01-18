import contextlib
import copy
import functools
import threading
import weakref
import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values
from tensorflow.python.distribute import values_util
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.saved_model import save_context
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
class PerWorkerVariable(resource_variable_ops.BaseResourceVariable):
    """A wrapper around unsynced variables created on workers.

  `PerWorkerVariable`s are variables that are stored on workers and not
  synchronized. A `PerWorkerVariable` is really a wrapper around multiple
  independent `Variable`s stored on independent worker machines. 
  `PerWorkerVariable` is currently only tested and supported when used with
  `ParameterServerStrategy`. A `PerWorkerVariable` can be created by creating a
  `Variable` within strategy scope and using the `per_worker_variable` flag,
  e.g.:

  ```
  with strategy.scope():
    var = tf.Variable(initial_value=0.0, per_worker_variable=True)
  ```

  The implementation modifies the graph to ensure that a worker's local version
  of the variable is used for computation at call time, while needing only one
  function trace and requiring no code changes beyond the `per_worker_variable`
  flag. `PerWorkerVariable`s can thus be treated like a standard `Variable`, but
  support is experimental and not all ops have been tested.

  All per-worker values can be retrieved and read into a list via
  `PerWorkerVariable.read_all()`.

  Caveats:
    - `PerWorkerVariable`s should not be used as direct inputs to a
      `tf.function`. That is, they should not appear in a tf.function header as
      an input argument. However they can still be read and manipulated in a
      `tf.function`.
    - The `shape` argument must be fully-defined (no `None` entries) or left
      empty. Partially-defined shapes are not yet supported.
    - Automatic control dependencies do not work with `PerWorkerVariable`s, so
      returning a `PerWorkerVariable` is not supported, and `read_all()` should 
      be used to retrieve values. (TODO: b/286052052)
    - `PerWorkerVariable`s should not be created within a `tf.function`.
  """

    def __init__(self, strategy, next_creator, **kwargs):
        self._coordinator = strategy._cluster_coordinator
        self._per_worker_vars = None
        self._var_creator = functools.partial(next_creator, **kwargs)
        self._coordinator_instance = next_creator(**kwargs)
        if kwargs.get('in_graph_mode') is None:
            with ops.init_scope():
                self._in_graph_mode = not context.executing_eagerly()
        else:
            self._in_graph_mode = kwargs['in_graph_mode']
        self._cached_value = None
        self._shape = self._coordinator_instance.shape
        self._dtype = self._coordinator_instance.dtype
        self._trainable = False
        self._unique_id = kwargs.get('unique_id')
        if kwargs.get('handle_name') is None:
            self._handle_name = 'Variable:0'
        else:
            self._handle_name = kwargs['handle_name'] + ':0'
        self._validate_shape = kwargs.get('validate_shape', True)

    @classmethod
    def _variable_call(cls, *args, **kwargs):
        """Override to be a no-op to avoid metaclass creating ResourceVariables."""
        return None

    @property
    def handle(self):
        if context.executing_eagerly() or save_context.in_save_context():
            return self._coordinator_instance.handle
        else:
            self._maybe_create_per_worker_vars()
            closure, spec = self.handle_call_time_value()
            return ops.get_default_graph().capture_call_time_value(closure, spec)

    def handle_call_time_value(self):
        """Returns a closure to run for a handle at call time and its spec.

    This function is called in self.handle to create a placeholder
    which returns a handle on some worker or on the coordinator.
    """

        def closure():
            dispatch_context = coordinator_context.get_current_dispatch_context()
            if dispatch_context:
                remote_value = self._per_worker_vars._values[dispatch_context.worker_index]
                ret = dispatch_context.maybe_get_remote_value(remote_value)
                return ret.handle
            else:
                return self._coordinator_instance.handle
        return (closure, PerWorkerVariableSpec(value=self._coordinator_instance.handle))

    def _maybe_create_per_worker_vars(self):
        """Create variable on each worker if it hasn't been created."""
        if not self._per_worker_vars:
            self._per_worker_vars = self._coordinator._create_per_worker_variables(self._var_creator)

    def read_all(self):
        """Synchronously read variables from all workers into a list of Tensors."""
        return [wv.get() for wv in self._per_worker_vars._values]