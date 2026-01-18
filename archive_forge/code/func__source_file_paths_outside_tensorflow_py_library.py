import socket
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import debug_pb2
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import tfprof_logger
def _source_file_paths_outside_tensorflow_py_library(code_defs, id_to_string):
    """Extract source file paths outside TensorFlow Python library.

  Args:
    code_defs: An iterable of `CodeDef` protos, i.e., an iterable of stack
      traces.
    id_to_string: A proto map from integer ids to strings.

  Returns:
    An iterable of source file paths outside the TensorFlow Python library.
  """
    file_ids = set()
    for code_def in code_defs:
        for trace in code_def.traces:
            file_ids.add(trace.file_id)
    non_tf_files = (id_to_string[file_id] for file_id in file_ids)
    non_tf_files = (f for f in non_tf_files if not source_utils.guess_is_tensorflow_py_library(f) and gfile.Exists(f))
    return non_tf_files