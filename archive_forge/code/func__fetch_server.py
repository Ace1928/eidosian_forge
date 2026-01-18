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
def _fetch_server(self, task_classes):
    endpoints = []
    for cls in task_classes:
        endpoints.append(endpoint.Endpoint(cls))
    server = worker_server.Server(TEST_TOPIC, TEST_EXCHANGE, futurist.ThreadPoolExecutor(max_workers=1), endpoints, transport='memory', transport_options={'polling_interval': POLLING_INTERVAL})
    server_thread = threading_utils.daemon_thread(server.start)
    return (server, server_thread)