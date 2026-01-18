import io
import logging
import queue
import threading
import uuid
import grpc
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray._private.ray_logging import global_worker_stdstream_dispatcher
from ray._private.worker import print_worker_logs
from ray.util.client.common import CLIENT_SERVER_MAX_THREADS
class StdStreamHandler:

    def __init__(self, queue):
        self.queue = queue
        self.id = str(uuid.uuid4())

    def handle(self, data):
        logdata = ray_client_pb2.LogData()
        logdata.level = -2 if data['is_err'] else -1
        logdata.name = 'stderr' if data['is_err'] else 'stdout'
        with io.StringIO() as file:
            print_worker_logs(data, file)
            logdata.msg = file.getvalue()
        self.queue.put(logdata)

    def register_global(self):
        global_worker_stdstream_dispatcher.add_handler(self.id, self.handle)

    def unregister_global(self):
        global_worker_stdstream_dispatcher.remove_handler(self.id)