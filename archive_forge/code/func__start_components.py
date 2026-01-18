import futurist
from futurist import waiters
from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor as base_executor
from taskflow.engines.worker_based import endpoint
from taskflow.engines.worker_based import executor as worker_executor
from taskflow.engines.worker_based import server as worker_server
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
from taskflow.utils import threading_utils
def _start_components(self, task_classes):
    server, server_thread = self._fetch_server(task_classes)
    executor = self._fetch_executor()
    self.addCleanup(executor.stop)
    self.addCleanup(server_thread.join)
    self.addCleanup(server.stop)
    executor.start()
    server_thread.start()
    server.wait()
    return (executor, server)