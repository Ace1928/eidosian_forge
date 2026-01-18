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
def _unary_response_in_pool(rpc_event: cygrpc.BaseEvent, state: _RPCState, behavior: ArityAgnosticMethodHandler, argument_thunk: Callable[[], Any], request_deserializer: Optional[SerializingFunction], response_serializer: Optional[SerializingFunction]) -> None:
    cygrpc.install_context_from_request_call_event(rpc_event)
    try:
        argument = argument_thunk()
        if argument is not None:
            response, proceed = _call_behavior(rpc_event, state, behavior, argument, request_deserializer)
            if proceed:
                serialized_response = _serialize_response(rpc_event, state, response, response_serializer)
                if serialized_response is not None:
                    _status(rpc_event, state, serialized_response)
    except Exception:
        traceback.print_exc()
    finally:
        cygrpc.uninstall_context()