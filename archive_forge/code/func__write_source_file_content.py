import atexit
import os
import re
import socket
import threading
import uuid
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
def _write_source_file_content(self, file_path):
    """Send the content of a source file via debug-events writer.

    Args:
      file_path: Path to the source file.

    Returns:
      An int index for the file.
    """
    if file_path in self._source_file_paths:
        return self._source_file_paths.index(file_path)
    with self._source_file_paths_lock:
        if file_path not in self._source_file_paths:
            lines = None
            if source_utils.is_extension_uncompiled_python_source(file_path):
                try:
                    lines, _ = source_utils.load_source(file_path)
                except IOError as e:
                    logging.warn('Failed to read source code from path: %s. Reason: %s', file_path, e)
            writer = self.get_writer()
            writer.WriteSourceFile(debug_event_pb2.SourceFile(file_path=file_path, host_name=self._hostname, lines=lines))
            self._source_file_paths.append(file_path)
        return self._source_file_paths.index(file_path)