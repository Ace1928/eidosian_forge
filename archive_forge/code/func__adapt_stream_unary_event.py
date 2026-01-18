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
def _adapt_stream_unary_event(stream_unary_event):

    def adaptation(request_iterator, servicer_context):
        callback = _Callback()
        if not servicer_context.add_callback(callback.cancel):
            raise abandonment.Abandoned()
        request_consumer = stream_unary_event(callback.consume_and_terminate, _FaceServicerContext(servicer_context))
        _run_request_pipe_thread(request_iterator, request_consumer, servicer_context)
        return callback.draw_all_values()[0]
    return adaptation