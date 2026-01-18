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
def _take_response_from_response_iterator(rpc_event: cygrpc.BaseEvent, state: _RPCState, response_iterator: Iterator[ResponseType]) -> Tuple[ResponseType, bool]:
    try:
        return (next(response_iterator), True)
    except StopIteration:
        return (None, True)
    except Exception as exception:
        with state.condition:
            if state.aborted:
                _abort(state, rpc_event.call, cygrpc.StatusCode.unknown, b'RPC Aborted')
            elif exception not in state.rpc_errors:
                details = 'Exception iterating responses: {}'.format(exception)
                _LOGGER.exception(details)
                _abort(state, rpc_event.call, cygrpc.StatusCode.unknown, _common.encode(details))
        return (None, False)