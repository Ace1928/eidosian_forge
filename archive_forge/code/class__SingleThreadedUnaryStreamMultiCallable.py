import copy
import functools
import logging
import os
import sys
import threading
import time
import types
from typing import (
import grpc  # pytype: disable=pyi-error
from grpc import _common  # pytype: disable=pyi-error
from grpc import _compression  # pytype: disable=pyi-error
from grpc import _grpcio_metadata  # pytype: disable=pyi-error
from grpc import _observability  # pytype: disable=pyi-error
from grpc._cython import cygrpc
from grpc._typing import ChannelArgumentType
from grpc._typing import DeserializingFunction
from grpc._typing import IntegratedCallFactory
from grpc._typing import MetadataType
from grpc._typing import NullaryCallbackType
from grpc._typing import ResponseType
from grpc._typing import SerializingFunction
from grpc._typing import UserTag
import grpc.experimental  # pytype: disable=pyi-error
class _SingleThreadedUnaryStreamMultiCallable(grpc.UnaryStreamMultiCallable):
    _channel: cygrpc.Channel
    _method: bytes
    _target: bytes
    _request_serializer: Optional[SerializingFunction]
    _response_deserializer: Optional[DeserializingFunction]
    _context: Any
    __slots__ = ['_channel', '_method', '_target', '_request_serializer', '_response_deserializer', '_context']

    def __init__(self, channel: cygrpc.Channel, method: bytes, target: bytes, request_serializer: SerializingFunction, response_deserializer: DeserializingFunction):
        self._channel = channel
        self._method = method
        self._target = target
        self._request_serializer = request_serializer
        self._response_deserializer = response_deserializer
        self._context = cygrpc.build_census_context()

    def __call__(self, request: Any, timeout: Optional[float]=None, metadata: Optional[MetadataType]=None, credentials: Optional[grpc.CallCredentials]=None, wait_for_ready: Optional[bool]=None, compression: Optional[grpc.Compression]=None) -> _SingleThreadedRendezvous:
        deadline = _deadline(timeout)
        serialized_request = _common.serialize(request, self._request_serializer)
        if serialized_request is None:
            state = _RPCState((), (), (), grpc.StatusCode.INTERNAL, 'Exception serializing request!')
            raise _InactiveRpcError(state)
        state = _RPCState(_UNARY_STREAM_INITIAL_DUE, None, None, None, None)
        call_credentials = None if credentials is None else credentials._credentials
        initial_metadata_flags = _InitialMetadataFlags().with_wait_for_ready(wait_for_ready)
        augmented_metadata = _compression.augment_metadata(metadata, compression)
        operations = ((cygrpc.SendInitialMetadataOperation(augmented_metadata, initial_metadata_flags), cygrpc.SendMessageOperation(serialized_request, _EMPTY_FLAGS), cygrpc.SendCloseFromClientOperation(_EMPTY_FLAGS)), (cygrpc.ReceiveStatusOnClientOperation(_EMPTY_FLAGS),), (cygrpc.ReceiveInitialMetadataOperation(_EMPTY_FLAGS),))
        operations_and_tags = tuple(((ops, None) for ops in operations))
        state.rpc_start_time = time.perf_counter()
        state.method = _common.decode(self._method)
        state.target = _common.decode(self._target)
        call = self._channel.segregated_call(cygrpc.PropagationConstants.GRPC_PROPAGATE_DEFAULTS, self._method, None, _determine_deadline(deadline), metadata, call_credentials, operations_and_tags, self._context)
        return _SingleThreadedRendezvous(state, call, self._response_deserializer, deadline)