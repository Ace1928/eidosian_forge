import tempfile
from tensorflow.core.protobuf import data_service_pb2
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
class _RemoteWorkerProcess(multi_process_lib.Process):
    """Runs a worker server in a new process to simulate a remote worker."""

    def __init__(self, dispatcher_address, port, worker_tags, pipe_writer):
        super(_RemoteWorkerProcess, self).__init__()
        self._dispatcher_address = dispatcher_address
        self._port = port
        self._worker_tags = worker_tags
        self._pipe_writer = pipe_writer

    def run(self):
        self.start_worker()

    def start_worker(self):
        self._worker = data_service_test_base.TestWorker(self._dispatcher_address, _WORKER_SHUTDOWN_QUIET_PERIOD_MS, port=self._port, worker_tags=self._worker_tags)
        self._worker.start()
        self._pipe_writer.send(self._worker.worker_address())
        self._worker.join()