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
def _consume_request_iterator(request_iterator: Iterator, state: _RPCState, call: Union[cygrpc.IntegratedCall, cygrpc.SegregatedCall], request_serializer: SerializingFunction, event_handler: Optional[UserTag]) -> None:
    """Consume a request supplied by the user."""

    def consume_request_iterator():
        while True:
            return_from_user_request_generator_invoked = False
            try:
                cygrpc.enter_user_request_generator()
                request = next(request_iterator)
            except StopIteration:
                break
            except Exception:
                cygrpc.return_from_user_request_generator()
                return_from_user_request_generator_invoked = True
                code = grpc.StatusCode.UNKNOWN
                details = 'Exception iterating requests!'
                _LOGGER.exception(details)
                call.cancel(_common.STATUS_CODE_TO_CYGRPC_STATUS_CODE[code], details)
                _abort(state, code, details)
                return
            finally:
                if not return_from_user_request_generator_invoked:
                    cygrpc.return_from_user_request_generator()
            serialized_request = _common.serialize(request, request_serializer)
            with state.condition:
                if state.code is None and (not state.cancelled):
                    if serialized_request is None:
                        code = grpc.StatusCode.INTERNAL
                        details = 'Exception serializing request!'
                        call.cancel(_common.STATUS_CODE_TO_CYGRPC_STATUS_CODE[code], details)
                        _abort(state, code, details)
                        return
                    else:
                        state.due.add(cygrpc.OperationType.send_message)
                        operations = (cygrpc.SendMessageOperation(serialized_request, _EMPTY_FLAGS),)
                        operating = call.operate(operations, event_handler)
                        if not operating:
                            state.due.remove(cygrpc.OperationType.send_message)
                            return

                        def _done():
                            return state.code is not None or cygrpc.OperationType.send_message not in state.due
                        _common.wait(state.condition.wait, _done, spin_cb=functools.partial(cygrpc.block_if_fork_in_progress, state))
                        if state.code is not None:
                            return
                else:
                    return
        with state.condition:
            if state.code is None:
                state.due.add(cygrpc.OperationType.send_close_from_client)
                operations = (cygrpc.SendCloseFromClientOperation(_EMPTY_FLAGS),)
                operating = call.operate(operations, event_handler)
                if not operating:
                    state.due.remove(cygrpc.OperationType.send_close_from_client)
    consumption_thread = cygrpc.ForkManagedThread(target=consume_request_iterator)
    consumption_thread.setDaemon(True)
    consumption_thread.start()