import socket
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import debug_pb2
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import tfprof_logger
def _load_debugged_source_file(file_path, source_file_proto):
    file_stat = gfile.Stat(file_path)
    source_file_proto.host = socket.gethostname()
    source_file_proto.file_path = file_path
    source_file_proto.last_modified = file_stat.mtime_nsec
    source_file_proto.bytes = file_stat.length
    try:
        with gfile.Open(file_path, 'r') as f:
            source_file_proto.lines.extend(f.read().splitlines())
    except IOError:
        pass