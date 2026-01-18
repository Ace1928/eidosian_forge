import collections
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
def _get_time_or_placeholder(value):
    """Modifies time-based config values to account for special behaviors."""
    if value == 0:
        return 1
    if value is None:
        return 0
    return value