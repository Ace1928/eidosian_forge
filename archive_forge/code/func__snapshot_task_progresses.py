import collections
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.data.experimental.service import _pywrap_utils
from tensorflow.python.util.tf_export import tf_export
def _snapshot_task_progresses(self):
    """Returns the progresses of the snapshot tasks currently being executed.

    Returns:
      An `Iterable[common_pb2.SnapshotTaskProgress]`.
    """
    return self._server.snapshot_task_progresses()