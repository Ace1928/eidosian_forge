import os.path
import time
import warnings
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import plugin_asset
from tensorflow.python.summary.writer.event_file_writer import EventFileWriter
from tensorflow.python.summary.writer.event_file_writer_v2 import EventFileWriterV2
from tensorflow.python.util.tf_export import tf_export
def add_event(self, event):
    """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
    self._warn_if_event_writer_is_closed()
    self.event_writer.add_event(event)