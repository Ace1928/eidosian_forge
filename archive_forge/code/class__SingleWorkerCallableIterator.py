from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.deprecation import deprecated
class _SingleWorkerCallableIterator(object):
    """Iterator for a single tensor-returning callable."""

    def __init__(self, fn, worker, devices):
        self._fn = fn
        self._worker = worker
        self._devices = devices

    def get_next(self, device, name=None):
        """Get next element for the given device from the callable."""
        del device, name
        with ops.device(self._worker):
            return self._fn()

    def get_next_as_list(self, name=None):
        """Get next element from the callable."""
        del name
        with ops.device(self._worker):
            data_list = [self._fn() for _ in self._devices]
            return data_list

    def get_next_as_optional_list(self):
        with ops.device(self._worker):
            data_list = [optional_ops.Optional.from_value(self._fn()) for _ in self._devices]
            return data_list

    def initialize(self):
        return []