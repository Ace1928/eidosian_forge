import tempfile
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
class TestWorker:
    """A tf.data service worker."""

    def __init__(self, dispatcher_address, shutdown_quiet_period_ms, protocol=PROTOCOL, data_transfer_protocol=None, port=0, worker_tags=None, cross_trainer_cache_size_bytes=None, snapshot_max_chunk_size_bytes=TEST_SNAPSHOT_MAX_CHUNK_SIZE_BYTES):
        self._dispatcher_address = dispatcher_address
        self._shutdown_quiet_period_ms = shutdown_quiet_period_ms
        self._server = _make_worker(dispatcher_address, protocol, data_transfer_protocol, shutdown_quiet_period_ms, port=port, worker_tags=worker_tags, cross_trainer_cache_size_bytes=cross_trainer_cache_size_bytes, snapshot_max_chunk_size_bytes=snapshot_max_chunk_size_bytes)
        self._running = False
        self._protocol = protocol
        self._data_transfer_protocol = data_transfer_protocol

    def stop(self):
        self._server._stop()
        self._running = False

    def start(self):
        self._server.start()
        self._port = int(self._server._address.split(':')[1])
        self._running = True

    def restart(self, use_same_port=True):
        """Restarts the worker, stopping it first if it is already running."""
        if self._running:
            self.stop()
        port = 0
        if use_same_port:
            port = self._port
        self._server = _make_worker(self._dispatcher_address, self._protocol, self._data_transfer_protocol, self._shutdown_quiet_period_ms, port)
        self._server.start()
        self._port = int(self._server._address.split(':')[1])
        self._running = True

    def join(self):
        self._server.join()

    def num_tasks(self):
        return self._server._num_tasks()

    def snapshot_task_progresses(self):
        return self._server._snapshot_task_progresses()

    def worker_address(self):
        return self._server._address