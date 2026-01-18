import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
def WriteExecution(self, execution):
    """Write a Execution proto with the writer.

    Args:
      execution: An Execution proto, describing a TensorFlow op or graph
        execution event.
    """
    debug_event = debug_event_pb2.DebugEvent(execution=execution)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteExecution(self._dump_root, debug_event)