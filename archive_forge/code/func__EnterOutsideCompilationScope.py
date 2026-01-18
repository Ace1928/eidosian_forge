from typing import Any, Callable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
def _EnterOutsideCompilationScope(self, cluster: Optional[Text]=None, is_map_outside_compilation=False):

    class FakeOp(object):
        """A helper class to determine the current device.

      Supports only the type and device set/get methods needed to run the
      graph's _apply_device_function method.
      """

        def __init__(self):
            self._device = ''

        @property
        def type(self):
            return 'FakeOp'

        @property
        def device(self):
            return self._device

        def _set_device(self, device):
            if isinstance(device, pydev.DeviceSpec):
                self._device = device.to_string()
            else:
                self._device = device

        def _set_device_from_string(self, device_str):
            self._device = device_str
    if self._outside_compilation_cluster:
        raise NotImplementedError('Cannot nest outside_compilation clusters')
    if cluster:
        self._outside_compilation_cluster = cluster
    else:
        self._outside_compilation_cluster = str(self._outside_compilation_counter)
        self._outside_compilation_counter += 1
    if is_map_outside_compilation:
        self._is_map_outside_compilation = True
    graph = ops.get_default_graph()
    fake_op = FakeOp()
    graph._apply_device_functions(fake_op)
    device = pydev.DeviceSpec.from_string(fake_op.device)
    if device.device_type == 'TPU_REPLICATED_CORE' and device.device_index is not None:
        self._host_compute_core.append(self._outside_compilation_cluster + ':' + str(device.device_index))
    self._oc_dev_fn_stack = graph._device_function_stack
    graph._device_function_stack = self._outer_device_function_stack