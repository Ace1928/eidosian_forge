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
class OutsideCompilationV2Context(control_flow_ops.ControlFlowContext):
    """The context for outside compilation in Tensorflow 2.0.

  Every op added in this context will be assigned an _xla_outside_compilation
  attribute.
  """

    def __init__(self, name: Text, is_map_outside_compilation=False):
        control_flow_ops.ControlFlowContext.__init__(self)
        self._name = name
        self._is_map_outside_compilation = is_map_outside_compilation

    def AddOp(self, op: ops.Operation) -> None:
        if self._outer_context:
            self._outer_context.AddOp(op)
        self._set_outside_compilation_attributes(op)

    def AddInnerOp(self, op: ops.Operation) -> None:
        if self._outer_context:
            self._outer_context.AddInnerOp(op)
        self._set_outside_compilation_attributes(op)

    def to_control_flow_context_def(self, context_def, export_scope=None):
        raise NotImplementedError

    def _set_outside_compilation_attributes(self, op: ops.Operation) -> None:
        op._set_attr(_OUTSIDE_COMPILATION_ATTR, attr_value_pb2.AttrValue(s=compat.as_bytes(self._name)))
        if self._is_map_outside_compilation:
            op._set_attr(_MAP_OUTSIDE_COMPILATION_ATTR, attr_value_pb2.AttrValue(b=True))