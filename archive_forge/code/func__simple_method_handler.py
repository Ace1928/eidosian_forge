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
def _simple_method_handler(implementation, request_deserializer, response_serializer):
    if implementation.style is style.Service.INLINE:
        if implementation.cardinality is cardinality.Cardinality.UNARY_UNARY:
            return _SimpleMethodHandler(False, False, request_deserializer, response_serializer, _adapt_unary_request_inline(implementation.unary_unary_inline), None, None, None)
        elif implementation.cardinality is cardinality.Cardinality.UNARY_STREAM:
            return _SimpleMethodHandler(False, True, request_deserializer, response_serializer, None, _adapt_unary_request_inline(implementation.unary_stream_inline), None, None)
        elif implementation.cardinality is cardinality.Cardinality.STREAM_UNARY:
            return _SimpleMethodHandler(True, False, request_deserializer, response_serializer, None, None, _adapt_stream_request_inline(implementation.stream_unary_inline), None)
        elif implementation.cardinality is cardinality.Cardinality.STREAM_STREAM:
            return _SimpleMethodHandler(True, True, request_deserializer, response_serializer, None, None, None, _adapt_stream_request_inline(implementation.stream_stream_inline))
    elif implementation.style is style.Service.EVENT:
        if implementation.cardinality is cardinality.Cardinality.UNARY_UNARY:
            return _SimpleMethodHandler(False, False, request_deserializer, response_serializer, _adapt_unary_unary_event(implementation.unary_unary_event), None, None, None)
        elif implementation.cardinality is cardinality.Cardinality.UNARY_STREAM:
            return _SimpleMethodHandler(False, True, request_deserializer, response_serializer, None, _adapt_unary_stream_event(implementation.unary_stream_event), None, None)
        elif implementation.cardinality is cardinality.Cardinality.STREAM_UNARY:
            return _SimpleMethodHandler(True, False, request_deserializer, response_serializer, None, None, _adapt_stream_unary_event(implementation.stream_unary_event), None)
        elif implementation.cardinality is cardinality.Cardinality.STREAM_STREAM:
            return _SimpleMethodHandler(True, True, request_deserializer, response_serializer, None, None, None, _adapt_stream_stream_event(implementation.stream_stream_event))
    raise ValueError()