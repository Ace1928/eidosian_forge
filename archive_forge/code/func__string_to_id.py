import socket
import grpc
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import debug_pb2
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import tfprof_logger
def _string_to_id(string, string_to_id):
    if string not in string_to_id:
        string_to_id[string] = len(string_to_id)
    return string_to_id[string]