import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _graph_execution_trace_digest_from_debug_event_proto(self, debug_event, locator):
    trace_proto = debug_event.graph_execution_trace
    op_name = trace_proto.op_name
    op_type = self._lookup_op_type(trace_proto.tfdbg_context_id, op_name)
    return GraphExecutionTraceDigest(debug_event.wall_time, locator, op_type, op_name, trace_proto.output_slot, debug_event.graph_execution_trace.tfdbg_context_id)