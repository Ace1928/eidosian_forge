import collections
import os
import threading
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat
def _execution_digest_from_debug_event_proto(debug_event, locator):
    """Convert a DebugEvent proto into an ExecutionDigest data object."""
    return ExecutionDigest(debug_event.wall_time, locator, debug_event.execution.op_type, output_tensor_device_ids=debug_event.execution.output_tensor_device_ids or None)