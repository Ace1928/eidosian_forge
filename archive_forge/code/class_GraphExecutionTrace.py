import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
class GraphExecutionTrace(GraphExecutionTraceDigest):
    """Detailed data object describing an intra-graph tensor execution.

  Attributes (in addition to GraphExecutionTraceDigest):
    graph_ids: The debugger-generated IDs of the graphs that enclose the
      executed op (tensor), ordered from the outermost to the innermost.
    graph_id: The debugger-generated ID of the innermost (immediately-enclosing)
      graph.
    tensor_debug_mode: TensorDebugMode enum value.
    debug_tensor_value: Debug tensor values (only for non-FULL_TENSOR
      tensor_debug_mode). A list of numbers. See the documentation of the
      TensorDebugModes for the semantics of the numbers.
    device_name: Device on which the tensor resides (if available)
  """

    def __init__(self, graph_execution_trace_digest, graph_ids, tensor_debug_mode, debug_tensor_value=None, device_name=None):
        super().__init__(graph_execution_trace_digest.wall_time, graph_execution_trace_digest.locator, graph_execution_trace_digest.op_type, graph_execution_trace_digest.op_name, graph_execution_trace_digest.output_slot, graph_execution_trace_digest.graph_id)
        self._graph_ids = tuple(graph_ids)
        self._tensor_debug_mode = tensor_debug_mode
        self._debug_tensor_value = debug_tensor_value
        self._device_name = device_name

    @property
    def graph_ids(self):
        return self._graph_ids

    @property
    def graph_id(self):
        return self._graph_ids[-1]

    @property
    def tensor_debug_mode(self):
        return self._tensor_debug_mode

    @property
    def debug_tensor_value(self):
        return _tuple_or_none(self._debug_tensor_value)

    @property
    def device_name(self):
        return self._device_name

    def to_json(self):
        output = super().to_json()
        output.update({'graph_ids': self.graph_ids, 'tensor_debug_mode': self.tensor_debug_mode, 'debug_tensor_value': self.debug_tensor_value, 'device_name': self.device_name})
        return output