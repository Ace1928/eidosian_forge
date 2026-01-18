import collections
import contextlib
import os
import re
import threading
import time
import weakref
from six.moves import queue
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class Closure(object):
    """Hold a function to be scheduled and its arguments."""

    def __init__(self, function, cancellation_mgr, args=None, kwargs=None):
        if not callable(function):
            raise ValueError('Function passed to `ClusterCoordinator.schedule` must be a callable object.')
        self._args = args or ()
        self._kwargs = kwargs or {}
        _disallow_remote_value_as_input(self._args)
        _disallow_remote_value_as_input(self._kwargs)
        if isinstance(function, def_function.Function):
            replica_args = _select_worker_slice(0, self._args)
            replica_kwargs = _select_worker_slice(0, self._kwargs)
            with metric_utils.monitored_timer('function_tracing', state_tracker=function._get_tracing_count):
                self._concrete_function = function.get_concrete_function(*nest.map_structure(_maybe_as_type_spec, replica_args), **nest.map_structure(_maybe_as_type_spec, replica_kwargs))
        elif isinstance(function, tf_function.ConcreteFunction):
            self._concrete_function = function
        if hasattr(self, '_concrete_function'):
            self._output_type_spec = func_graph.convert_structure_to_signature(self._concrete_function.structured_outputs)
            self._function = cancellation_mgr.get_cancelable_function(self._concrete_function)
        else:
            self._output_type_spec = None
            self._function = function
        self._output_remote_value_ref = None

    def build_output_remote_value(self):
        if self._output_remote_value_ref is None:
            ret = RemoteValueImpl(None, self._output_type_spec)
            self._output_remote_value_ref = weakref.ref(ret)
            return ret
        else:
            raise ValueError('The output of the Closure cannot be built more than once.')

    def maybe_call_with_output_remote_value(self, method):
        if self._output_remote_value_ref is None:
            return None
        output_remote_value = self._output_remote_value_ref()
        if output_remote_value is not None:
            return method(output_remote_value)
        return None

    def mark_cancelled(self):
        e = errors.CancelledError(None, None, 'The corresponding function is cancelled. Please reschedule the function.')
        self.maybe_call_with_output_remote_value(lambda r: r._set_error(e))

    def execute_on(self, worker):
        """Executes the closure on the given worker.

    Args:
      worker: a `Worker` object.
    """
        replica_args = _select_worker_slice(worker.worker_index, self._args)
        replica_kwargs = _select_worker_slice(worker.worker_index, self._kwargs)
        e = _get_error_from_remote_values(replica_args) or _get_error_from_remote_values(replica_kwargs)
        if e:
            if not isinstance(e, ClosureInputError):
                e = ClosureInputError(e)
            raise e
        with ops.device(worker.device_name):
            with context.executor_scope(worker.executor):
                with coordinator_context.with_dispatch_context(worker):
                    with metric_utils.monitored_timer('closure_execution'):
                        output_values = self._function(*nest.map_structure(coordinator_context.maybe_get_remote_value, replica_args), **nest.map_structure(coordinator_context.maybe_get_remote_value, replica_kwargs))
        self.maybe_call_with_output_remote_value(lambda r: r._set_values(output_values))