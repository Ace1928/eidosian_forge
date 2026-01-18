import threading
import time
from google.protobuf import text_format
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
def _reset_manager(self, stopping=False):
    """Reset the graph, session and worker manager."""
    self._graph = ops.Graph()
    self._session = session_lib.Session(target=self._target, graph=self._graph, config=self._config)
    if self._devices is None:
        self._devices = all_worker_devices(self._session)
    with self._graph.as_default():
        self._worker_manager = WorkerHeartbeatManager.from_devices(self._session, self._devices)
    if stopping:
        timeout_ms = -1
        shutdown_mode = event_pb2.NOT_CONFIGURED
    else:
        timeout_ms = self.shutdown_timeout * 1000
        shutdown_mode = event_pb2.WAIT_FOR_COORDINATOR
    self._worker_manager.configure(event_pb2.WorkerHeartbeatRequest(watchdog_config=event_pb2.WatchdogConfig(timeout_ms=timeout_ms), shutdown_mode=shutdown_mode))