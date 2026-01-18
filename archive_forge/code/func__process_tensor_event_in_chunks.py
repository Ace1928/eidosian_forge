import collections
import json
import queue
import threading
import time
from concurrent import futures
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
def _process_tensor_event_in_chunks(self, event, tensor_chunks):
    """Possibly reassemble event chunks.

    Due to gRPC's message size limit, a large tensor can be encapsulated in
    multiple Event proto chunks to be sent through the debugger stream. This
    method keeps track of the chunks that have arrived, reassemble all chunks
    corresponding to a tensor when they have arrived and return the reassembled
    Event proto.

    Args:
      event: The single Event proto that has arrived.
      tensor_chunks: A dict used to keep track of the Event protos that have
        arrived but haven't been reassembled.

    Returns:
      If all Event protos corresponding to a tensor have arrived, returns the
      reassembled Event proto. Otherwise, return None.
    """
    value = event.summary.value[0]
    debugger_plugin_metadata = json.loads(compat.as_text(value.metadata.plugin_data.content))
    device_name = debugger_plugin_metadata['device']
    num_chunks = debugger_plugin_metadata['numChunks']
    chunk_index = debugger_plugin_metadata['chunkIndex']
    if num_chunks <= 1:
        return event
    debug_node_name = value.node_name
    timestamp = int(event.wall_time)
    tensor_key = '%s_%s_%d' % (device_name, debug_node_name, timestamp)
    if tensor_key not in tensor_chunks:
        tensor_chunks[tensor_key] = [None] * num_chunks
    chunks = tensor_chunks[tensor_key]
    if value.tensor.tensor_content:
        chunks[chunk_index] = value.tensor
    elif value.tensor.string_val:
        chunks[chunk_index] = event
    if None not in chunks:
        if value.tensor.tensor_content:
            event.summary.value[0].tensor.tensor_content = b''.join((chunk.tensor_content for chunk in chunks))
            del tensor_chunks[tensor_key]
            return event
        elif value.tensor.string_val:
            merged_event = chunks[0]
            for chunk in chunks[1:]:
                merged_event.summary.value[0].tensor.string_val.extend(list(chunk.summary.value[0].tensor.string_val))
            return merged_event