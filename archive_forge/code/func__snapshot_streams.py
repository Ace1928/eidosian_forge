import collections
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
def _snapshot_streams(self, path):
    """Returns information about all the streams for a snapshot."""
    return self._server.snapshot_streams(path)