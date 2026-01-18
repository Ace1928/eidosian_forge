import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _graph_execution_trace_from_debug_event_proto(self, debug_event, locator):
    """Convert a DebugEvent proto into a GraphExecutionTrace data object."""
    trace_proto = debug_event.graph_execution_trace
    graph_ids = [trace_proto.tfdbg_context_id]
    while True:
        graph = self.graph_by_id(graph_ids[0])
        if graph.outer_graph_id:
            graph_ids.insert(0, graph.outer_graph_id)
        else:
            break
    if trace_proto.tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
        debug_tensor_value = None
    else:
        debug_tensor_value = _parse_tensor_value(trace_proto.tensor_proto, return_list=True)
    return GraphExecutionTrace(self._graph_execution_trace_digest_from_debug_event_proto(debug_event, locator), graph_ids=graph_ids, tensor_debug_mode=trace_proto.tensor_debug_mode, debug_tensor_value=debug_tensor_value, device_name=trace_proto.device_name or None)