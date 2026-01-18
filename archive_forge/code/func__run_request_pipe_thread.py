import collections
import threading
import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.common import style
from grpc.framework.foundation import abandonment
from grpc.framework.foundation import logging_pool
from grpc.framework.foundation import stream
from grpc.framework.interfaces.face import face
def _run_request_pipe_thread(request_iterator, request_consumer, servicer_context):
    thread_joined = threading.Event()

    def pipe_requests():
        for request in request_iterator:
            if not servicer_context.is_active() or thread_joined.is_set():
                return
            request_consumer.consume(request)
            if not servicer_context.is_active() or thread_joined.is_set():
                return
        request_consumer.terminate()
    request_pipe_thread = threading.Thread(target=pipe_requests)
    request_pipe_thread.daemon = True
    request_pipe_thread.start()