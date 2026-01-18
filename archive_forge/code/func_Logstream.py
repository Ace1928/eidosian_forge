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
def Logstream(self, request_iterator, context):
    initialized = False
    with self.client_lock:
        threshold = CLIENT_SERVER_MAX_THREADS / 2
        if self.num_clients + 1 >= threshold:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            logger.warning(f'Logstream: Num clients {self.num_clients} has reached the threshold {threshold}. Rejecting new connection.')
            return
        self.num_clients += 1
        initialized = True
        logger.info(f'New logs connection established. Total clients: {self.num_clients}')
    log_queue = queue.Queue()
    thread = threading.Thread(target=log_status_change_thread, args=(log_queue, request_iterator), daemon=True)
    thread.start()
    try:
        queue_iter = iter(log_queue.get, None)
        for record in queue_iter:
            if record is None:
                break
            yield record
    except grpc.RpcError as e:
        logger.debug(f'Closing log channel: {e}')
    finally:
        thread.join()
        with self.client_lock:
            if initialized:
                self.num_clients -= 1