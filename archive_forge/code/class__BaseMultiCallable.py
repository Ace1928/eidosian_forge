import asyncio
import sys
from typing import Any, Iterable, List, Optional, Sequence
import grpc
from grpc import _common
from grpc import _compression
from grpc import _grpcio_metadata
from grpc._cython import cygrpc
from . import _base_call
from . import _base_channel
from ._call import StreamStreamCall
from ._call import StreamUnaryCall
from ._call import UnaryStreamCall
from ._call import UnaryUnaryCall
from ._interceptor import ClientInterceptor
from ._interceptor import InterceptedStreamStreamCall
from ._interceptor import InterceptedStreamUnaryCall
from ._interceptor import InterceptedUnaryStreamCall
from ._interceptor import InterceptedUnaryUnaryCall
from ._interceptor import StreamStreamClientInterceptor
from ._interceptor import StreamUnaryClientInterceptor
from ._interceptor import UnaryStreamClientInterceptor
from ._interceptor import UnaryUnaryClientInterceptor
from ._metadata import Metadata
from ._typing import ChannelArgumentType
from ._typing import DeserializingFunction
from ._typing import MetadataType
from ._typing import RequestIterableType
from ._typing import RequestType
from ._typing import ResponseType
from ._typing import SerializingFunction
from ._utils import _timeout_to_deadline
class _BaseMultiCallable:
    """Base class of all multi callable objects.

    Handles the initialization logic and stores common attributes.
    """
    _loop: asyncio.AbstractEventLoop
    _channel: cygrpc.AioChannel
    _method: bytes
    _request_serializer: SerializingFunction
    _response_deserializer: DeserializingFunction
    _interceptors: Optional[Sequence[ClientInterceptor]]
    _references: List[Any]
    _loop: asyncio.AbstractEventLoop

    def __init__(self, channel: cygrpc.AioChannel, method: bytes, request_serializer: SerializingFunction, response_deserializer: DeserializingFunction, interceptors: Optional[Sequence[ClientInterceptor]], references: List[Any], loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._channel = channel
        self._method = method
        self._request_serializer = request_serializer
        self._response_deserializer = response_deserializer
        self._interceptors = interceptors
        self._references = references

    @staticmethod
    def _init_metadata(metadata: Optional[MetadataType]=None, compression: Optional[grpc.Compression]=None) -> Metadata:
        """Based on the provided values for <metadata> or <compression> initialise the final
        metadata, as it should be used for the current call.
        """
        metadata = metadata or Metadata()
        if not isinstance(metadata, Metadata) and isinstance(metadata, tuple):
            metadata = Metadata.from_tuple(metadata)
        if compression:
            metadata = Metadata(*_compression.augment_metadata(metadata, compression))
        return metadata