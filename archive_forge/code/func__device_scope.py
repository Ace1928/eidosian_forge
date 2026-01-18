import copy
import weakref
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import tpu_util
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
def _device_scope(self):
    if self._packed_handle is None or values_util.is_saving_non_distributed() or tpu_util.enclosing_tpu_context() is not None:
        return ops.NullContextmanager()
    device = device_util.canonicalize(device_util.current())
    if device in self._device_to_handle:
        return ops.NullContextmanager()
    return ops.device(self._primary_handle.device)