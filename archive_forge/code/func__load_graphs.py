import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _load_graphs(self):
    """Incrementally read the .graphs file.

    Compiles the DebuggedGraph and GraphOpCreation data.
    """
    graphs_iter = self._reader.graphs_iterator()
    for debug_event, offset in graphs_iter:
        if debug_event.graph_op_creation.ByteSize():
            op_creation_proto = debug_event.graph_op_creation
            op_digest = GraphOpCreationDigest(debug_event.wall_time, offset, op_creation_proto.graph_id, op_creation_proto.op_type, op_creation_proto.op_name, tuple(op_creation_proto.output_tensor_ids), op_creation_proto.code_location.host_name, tuple(op_creation_proto.code_location.stack_frame_ids), input_names=tuple(op_creation_proto.input_names))
            self._graph_op_digests.append(op_digest)
            debugged_graph = self._graph_by_id[op_creation_proto.graph_id]
            debugged_graph.add_op(op_digest)
            for dst_slot, input_name in enumerate(op_creation_proto.input_names):
                src_op_name, src_slot = input_name.split(':')
                debugged_graph.add_op_consumer(src_op_name, int(src_slot), op_creation_proto.op_name, dst_slot)
        elif debug_event.debugged_graph.ByteSize():
            graph_proto = debug_event.debugged_graph
            graph = DebuggedGraph(graph_proto.graph_name or None, graph_proto.graph_id, outer_graph_id=graph_proto.outer_context_id or None)
            self._graph_by_id[graph_proto.graph_id] = graph
            if graph_proto.outer_context_id:
                self._graph_by_id[graph_proto.outer_context_id].add_inner_graph_id(graph.graph_id)
        elif debug_event.debugged_device.ByteSize():
            device_proto = debug_event.debugged_device
            self._device_by_id[device_proto.device_id] = DebuggedDevice(device_proto.device_name, device_proto.device_id)