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
def _process_encoded_graph_def_in_chunks(self, event, graph_def_chunks):
    """Process an Event proto containing a chunk of encoded GraphDef.

    Args:
      event: the Event proto containing the chunk of encoded GraphDef.
      graph_def_chunks: A dict mapping keys for GraphDefs (i.e.,
      "<graph_def_hash>,<device_name>,<wall_time>") to a list of chunks of
      encoded GraphDefs.

    Returns:
      If all chunks of the GraphDef have arrived,
        return decoded GraphDef proto, device name, wall_time.
      Otherwise,
        return None, None, None.
    """
    graph_def = graph_pb2.GraphDef()
    index_bar_0 = event.graph_def.find(b'|')
    index_bar_1 = event.graph_def.find(b'|', index_bar_0 + 1)
    index_bar_2 = event.graph_def.find(b'|', index_bar_1 + 1)
    graph_def_hash_device_timestamp = event.graph_def[:index_bar_0]
    chunk_index = int(event.graph_def[index_bar_0 + 1:index_bar_1])
    num_chunks = int(event.graph_def[index_bar_1 + 1:index_bar_2])
    if graph_def_hash_device_timestamp not in graph_def_chunks:
        graph_def_chunks[graph_def_hash_device_timestamp] = [None] * num_chunks
    graph_def_chunks[graph_def_hash_device_timestamp][chunk_index] = event.graph_def[index_bar_2 + 1:]
    if all(graph_def_chunks[graph_def_hash_device_timestamp]):
        device_name = graph_def_hash_device_timestamp.split(b',')[1]
        wall_time = int(graph_def_hash_device_timestamp.split(b',')[2])
        graph_def.ParseFromString(b''.join(graph_def_chunks[graph_def_hash_device_timestamp]))
        del graph_def_chunks[graph_def_hash_device_timestamp]
        self._process_graph_def(graph_def)
        return (graph_def, device_name, wall_time)
    else:
        return (None, None, None)