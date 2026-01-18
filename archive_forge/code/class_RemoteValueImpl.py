import threading
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec as type_spec_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
class RemoteValueImpl(remote_value.RemoteValue):
    """Implementation of `RemoteValue`."""

    def __init__(self, closure, type_spec):
        """Initializes a `RemoteValueImpl`.

    Args:
      closure: The closure from which the `RemoteValue` is created.
      type_spec: The type spec for this `RemoteValue` which is used to trace
        functions that take this `RemoteValue` as input.
    """
        self._closure = closure
        self._type_spec = type_spec
        self._values = None
        self._has_fetched_to_local = False
        self._has_fetched_to_local_lock = threading.Lock()
        self._fetched_tensors = None
        self._error = None
        self._status_available_event = threading.Event()
        self._status = remote_value.RemoteValueStatus.NOT_READY

    def _set_aborted(self, error):
        self._status = remote_value.RemoteValueStatus.ABORTED
        self._values = None
        self._error = error
        self._status_available_event.set()

    def _rebuild_on(self, worker):
        self._status_available_event.clear()
        self._closure.execute_on(worker)

    def _set_values(self, tensors):
        self._status = remote_value.RemoteValueStatus.READY
        self._values = tensors
        self._error = None
        self._status_available_event.set()

    def _set_error(self, error):
        self._status = remote_value.RemoteValueStatus.READY
        self._values = None
        self._error = error
        self._status_available_event.set()

    def _get_values(self):
        self._status_available_event.wait()
        return self._values

    def _get_error(self):
        self._status_available_event.wait()
        return self._error

    def _wait_and_maybe_error(self):
        self._status_available_event.wait()
        if self._status is remote_value.RemoteValueStatus.ABORTED:
            raise errors.CancelledError(None, None, 'The corresponding function is aborted. Please reschedule the function.')
        if self._error is not None:
            raise self._error

    def fetch(self):
        return nest.map_structure(lambda x: x.numpy() if hasattr(x, 'numpy') else x, self.get())

    def _copy_to_local(self):

        def copy_tensor(composite_tensor_obj):
            """Copy a remote tensor to local (coordinator)."""
            if isinstance(composite_tensor_obj, input_lib.DistributedIterator):
                return composite_tensor_obj
            with ops.device('/job:%s' % context.get_server_def().job_name):
                return array_ops.identity(composite_tensor_obj)
        fetched_result = None
        if self._values is not None:
            fetched_result = nest.map_structure(copy_tensor, self._values)
        return fetched_result

    def get(self):
        self._wait_and_maybe_error()
        with self._has_fetched_to_local_lock:
            if not self._has_fetched_to_local:
                self._fetched_tensors = self._copy_to_local()
                self._has_fetched_to_local = True
        return self._fetched_tensors