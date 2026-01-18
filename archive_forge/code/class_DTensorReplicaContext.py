from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
class DTensorReplicaContext(distribute_lib.ReplicaContext):
    """ReplicaContext for strategy that is backed by DTensor.

  Since the DTensor is operated in the global context, most of the methods from
  existing strategy ReplicaContext is not applicable since they need to access
  local values. For now most of the methods in this class will raise explicit
  error to user, and we will add more support for local values in future.
  """
    _UNSUPPORTED_ERROR_MSG = "Strategy that is backed by DTensor is run with a global context, and doesn't support operations for local context, like any call to merge/gather/reduce or local replica ID. Please use any strategy that is not backed by DTensor"

    def __init__(self, strategy):
        super().__init__(strategy, replica_id_in_sync_group=None)

    def __enter__(self):
        distribute_lib._push_per_thread_mode(self._thread_context)
        summary_state = summary_ops_v2._summary_state
        self._summary_recording_distribution_strategy = summary_state.is_recording_distribution_strategy
        summary_state.is_recording_distribution_strategy = True

    @property
    def replica_id_in_sync_group(self):
        return 0

    @property
    def _replica_id(self):
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

    def merge_call(self, merge_fn, args=(), kwargs=None):
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

    def all_reduce(self, reduce_op, value, options=None):
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

    def all_gather(self, value, axis, options=None):
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)

    def _update(self, var, fn, args=(), kwargs=None, group=True):
        raise NotImplementedError(self._UNSUPPORTED_ERROR_MSG)