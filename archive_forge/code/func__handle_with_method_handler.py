from __future__ import annotations
import collections
from concurrent import futures
import contextvars
import enum
import logging
import threading
import time
import traceback
from typing import (
import grpc  # pytype: disable=pyi-error
from grpc import _common  # pytype: disable=pyi-error
from grpc import _compression  # pytype: disable=pyi-error
from grpc import _interceptor  # pytype: disable=pyi-error
from grpc._cython import cygrpc
from grpc._typing import ArityAgnosticMethodHandler
from grpc._typing import ChannelArgumentType
from grpc._typing import DeserializingFunction
from grpc._typing import MetadataType
from grpc._typing import NullaryCallbackType
from grpc._typing import ResponseType
from grpc._typing import SerializingFunction
from grpc._typing import ServerCallbackTag
from grpc._typing import ServerTagCallbackType
def _handle_with_method_handler(rpc_event: cygrpc.BaseEvent, state: _RPCState, method_handler: grpc.RpcMethodHandler, thread_pool: futures.ThreadPoolExecutor) -> futures.Future:
    with state.condition:
        rpc_event.call.start_server_batch((cygrpc.ReceiveCloseOnServerOperation(_EMPTY_FLAGS),), _receive_close_on_server(state))
        state.due.add(_RECEIVE_CLOSE_ON_SERVER_TOKEN)
        if method_handler.request_streaming:
            if method_handler.response_streaming:
                return _handle_stream_stream(rpc_event, state, method_handler, thread_pool)
            else:
                return _handle_stream_unary(rpc_event, state, method_handler, thread_pool)
        elif method_handler.response_streaming:
            return _handle_unary_stream(rpc_event, state, method_handler, thread_pool)
        else:
            return _handle_unary_unary(rpc_event, state, method_handler, thread_pool)