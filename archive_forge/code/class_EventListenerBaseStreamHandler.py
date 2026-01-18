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
class EventListenerBaseStreamHandler:
    """Per-stream handler of EventListener gRPC streams."""

    def __init__(self):
        """Constructor of EventListenerBaseStreamHandler."""

    def on_core_metadata_event(self, event):
        """Callback for core metadata.

    Args:
      event: The Event proto that carries a JSON string in its
        `log_message.message` field.

    Returns:
      `None` or an `EventReply` proto to be sent back to the client. If `None`,
      an `EventReply` proto construct with the default no-arg constructor will
      be sent back to the client.
    """
        raise NotImplementedError('on_core_metadata_event() is not implemented in the base servicer class')

    def on_graph_def(self, graph_def, device_name, wall_time):
        """Callback for Event proto received through the gRPC stream.

    This Event proto carries a GraphDef, encoded as bytes, in its graph_def
    field.

    Args:
      graph_def: A GraphDef object.
      device_name: Name of the device on which the graph was created.
      wall_time: An epoch timestamp (in microseconds) for the graph.

    Returns:
      `None` or an `EventReply` proto to be sent back to the client. If `None`,
      an `EventReply` proto construct with the default no-arg constructor will
      be sent back to the client.
    """
        raise NotImplementedError('on_graph_def() is not implemented in the base servicer class')

    def on_value_event(self, event):
        """Callback for Event proto received through the gRPC stream.

    This Event proto carries a Tensor in its summary.value[0] field.

    Args:
      event: The Event proto from the stream to be processed.
    """
        raise NotImplementedError('on_value_event() is not implemented in the base servicer class')