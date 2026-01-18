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
def _find_method_handler(rpc_event: cygrpc.BaseEvent, state: _RPCState, generic_handlers: List[grpc.GenericRpcHandler], interceptor_pipeline: Optional[_interceptor._ServicePipeline]) -> Optional[grpc.RpcMethodHandler]:

    def query_handlers(handler_call_details: _HandlerCallDetails) -> Optional[grpc.RpcMethodHandler]:
        for generic_handler in generic_handlers:
            method_handler = generic_handler.service(handler_call_details)
            if method_handler is not None:
                return method_handler
        return None
    handler_call_details = _HandlerCallDetails(_common.decode(rpc_event.call_details.method), rpc_event.invocation_metadata)
    if interceptor_pipeline is not None:
        return state.context.run(interceptor_pipeline.execute, query_handlers, handler_call_details)
    else:
        return state.context.run(query_handlers, handler_call_details)