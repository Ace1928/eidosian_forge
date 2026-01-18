import grpc
from grpc import _common
from grpc.beta import _metadata
from grpc.beta import interfaces
from grpc.framework.common import cardinality
from grpc.framework.foundation import future
from grpc.framework.interfaces.face import face
class _GenericStub(face.GenericStub):

    def __init__(self, channel, metadata_transformer, request_serializers, response_deserializers):
        self._channel = channel
        self._metadata_transformer = metadata_transformer
        self._request_serializers = request_serializers or {}
        self._response_deserializers = response_deserializers or {}

    def blocking_unary_unary(self, group, method, request, timeout, metadata=None, with_call=None, protocol_options=None):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _blocking_unary_unary(self._channel, group, method, timeout, with_call, protocol_options, metadata, self._metadata_transformer, request, request_serializer, response_deserializer)

    def future_unary_unary(self, group, method, request, timeout, metadata=None, protocol_options=None):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _future_unary_unary(self._channel, group, method, timeout, protocol_options, metadata, self._metadata_transformer, request, request_serializer, response_deserializer)

    def inline_unary_stream(self, group, method, request, timeout, metadata=None, protocol_options=None):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _unary_stream(self._channel, group, method, timeout, protocol_options, metadata, self._metadata_transformer, request, request_serializer, response_deserializer)

    def blocking_stream_unary(self, group, method, request_iterator, timeout, metadata=None, with_call=None, protocol_options=None):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _blocking_stream_unary(self._channel, group, method, timeout, with_call, protocol_options, metadata, self._metadata_transformer, request_iterator, request_serializer, response_deserializer)

    def future_stream_unary(self, group, method, request_iterator, timeout, metadata=None, protocol_options=None):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _future_stream_unary(self._channel, group, method, timeout, protocol_options, metadata, self._metadata_transformer, request_iterator, request_serializer, response_deserializer)

    def inline_stream_stream(self, group, method, request_iterator, timeout, metadata=None, protocol_options=None):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _stream_stream(self._channel, group, method, timeout, protocol_options, metadata, self._metadata_transformer, request_iterator, request_serializer, response_deserializer)

    def event_unary_unary(self, group, method, request, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        raise NotImplementedError()

    def event_unary_stream(self, group, method, request, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        raise NotImplementedError()

    def event_stream_unary(self, group, method, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        raise NotImplementedError()

    def event_stream_stream(self, group, method, receiver, abortion_callback, timeout, metadata=None, protocol_options=None):
        raise NotImplementedError()

    def unary_unary(self, group, method):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _UnaryUnaryMultiCallable(self._channel, group, method, self._metadata_transformer, request_serializer, response_deserializer)

    def unary_stream(self, group, method):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _UnaryStreamMultiCallable(self._channel, group, method, self._metadata_transformer, request_serializer, response_deserializer)

    def stream_unary(self, group, method):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _StreamUnaryMultiCallable(self._channel, group, method, self._metadata_transformer, request_serializer, response_deserializer)

    def stream_stream(self, group, method):
        request_serializer = self._request_serializers.get((group, method))
        response_deserializer = self._response_deserializers.get((group, method))
        return _StreamStreamMultiCallable(self._channel, group, method, self._metadata_transformer, request_serializer, response_deserializer)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False