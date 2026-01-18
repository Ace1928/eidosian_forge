import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
def FlushNonExecutionFiles(self):
    """Flush the non-execution debug event files."""
    _pywrap_debug_events_writer.FlushNonExecutionFiles(self._dump_root)