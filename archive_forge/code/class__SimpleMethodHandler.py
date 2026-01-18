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
class _SimpleMethodHandler(collections.namedtuple('_MethodHandler', ('request_streaming', 'response_streaming', 'request_deserializer', 'response_serializer', 'unary_unary', 'unary_stream', 'stream_unary', 'stream_stream')), grpc.RpcMethodHandler):
    pass