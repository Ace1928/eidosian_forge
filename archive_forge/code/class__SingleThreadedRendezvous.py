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
class _SingleThreadedRendezvous(_Rendezvous, grpc.Call, grpc.Future):
    """An RPC iterator operating entirely on a single thread.

    The __next__ method of _SingleThreadedRendezvous does not depend on the
    existence of any other thread, including the "channel spin thread".
    However, this means that its interface is entirely synchronous. So this
    class cannot completely fulfill the grpc.Future interface. The result,
    exception, and traceback methods will never block and will instead raise
    an exception if calling the method would result in blocking.

    This means that these methods are safe to call from add_done_callback
    handlers.
    """
    _state: _RPCState

    def _is_complete(self) -> bool:
        return self._state.code is not None

    def cancelled(self) -> bool:
        with self._state.condition:
            return self._state.cancelled

    def running(self) -> bool:
        with self._state.condition:
            return self._state.code is None

    def done(self) -> bool:
        with self._state.condition:
            return self._state.code is not None

    def result(self, timeout: Optional[float]=None) -> Any:
        """Returns the result of the computation or raises its exception.

        This method will never block. Instead, it will raise an exception
        if calling this method would otherwise result in blocking.

        Since this method will never block, any `timeout` argument passed will
        be ignored.
        """
        del timeout
        with self._state.condition:
            if not self._is_complete():
                raise grpc.experimental.UsageError('_SingleThreadedRendezvous only supports result() when the RPC is complete.')
            if self._state.code is grpc.StatusCode.OK:
                return self._state.response
            elif self._state.cancelled:
                raise grpc.FutureCancelledError()
            else:
                raise self

    def exception(self, timeout: Optional[float]=None) -> Optional[Exception]:
        """Return the exception raised by the computation.

        This method will never block. Instead, it will raise an exception
        if calling this method would otherwise result in blocking.

        Since this method will never block, any `timeout` argument passed will
        be ignored.
        """
        del timeout
        with self._state.condition:
            if not self._is_complete():
                raise grpc.experimental.UsageError('_SingleThreadedRendezvous only supports exception() when the RPC is complete.')
            if self._state.code is grpc.StatusCode.OK:
                return None
            elif self._state.cancelled:
                raise grpc.FutureCancelledError()
            else:
                return self

    def traceback(self, timeout: Optional[float]=None) -> Optional[types.TracebackType]:
        """Access the traceback of the exception raised by the computation.

        This method will never block. Instead, it will raise an exception
        if calling this method would otherwise result in blocking.

        Since this method will never block, any `timeout` argument passed will
        be ignored.
        """
        del timeout
        with self._state.condition:
            if not self._is_complete():
                raise grpc.experimental.UsageError('_SingleThreadedRendezvous only supports traceback() when the RPC is complete.')
            if self._state.code is grpc.StatusCode.OK:
                return None
            elif self._state.cancelled:
                raise grpc.FutureCancelledError()
            else:
                try:
                    raise self
                except grpc.RpcError:
                    return sys.exc_info()[2]

    def add_done_callback(self, fn: Callable[[grpc.Future], None]) -> None:
        with self._state.condition:
            if self._state.code is None:
                self._state.callbacks.append(functools.partial(fn, self))
                return
        fn(self)

    def initial_metadata(self) -> Optional[MetadataType]:
        """See grpc.Call.initial_metadata"""
        with self._state.condition:
            while self._state.initial_metadata is None:
                self._consume_next_event()
            return self._state.initial_metadata

    def trailing_metadata(self) -> Optional[MetadataType]:
        """See grpc.Call.trailing_metadata"""
        with self._state.condition:
            if self._state.trailing_metadata is None:
                raise grpc.experimental.UsageError('Cannot get trailing metadata until RPC is completed.')
            return self._state.trailing_metadata

    def code(self) -> Optional[grpc.StatusCode]:
        """See grpc.Call.code"""
        with self._state.condition:
            if self._state.code is None:
                raise grpc.experimental.UsageError('Cannot get code until RPC is completed.')
            return self._state.code

    def details(self) -> Optional[str]:
        """See grpc.Call.details"""
        with self._state.condition:
            if self._state.details is None:
                raise grpc.experimental.UsageError('Cannot get details until RPC is completed.')
            return _common.decode(self._state.details)

    def _consume_next_event(self) -> Optional[cygrpc.BaseEvent]:
        event = self._call.next_event()
        with self._state.condition:
            callbacks = _handle_event(event, self._state, self._response_deserializer)
            for callback in callbacks:
                callback()
        return event

    def _next_response(self) -> Any:
        while True:
            self._consume_next_event()
            with self._state.condition:
                if self._state.response is not None:
                    response = self._state.response
                    self._state.response = None
                    return response
                elif cygrpc.OperationType.receive_message not in self._state.due:
                    if self._state.code is grpc.StatusCode.OK:
                        raise StopIteration()
                    elif self._state.code is not None:
                        raise self

    def _next(self) -> Any:
        with self._state.condition:
            if self._state.code is None:
                self._state.due.add(cygrpc.OperationType.receive_message)
                operating = self._call.operate((cygrpc.ReceiveMessageOperation(_EMPTY_FLAGS),), None)
                if not operating:
                    self._state.due.remove(cygrpc.OperationType.receive_message)
            elif self._state.code is grpc.StatusCode.OK:
                raise StopIteration()
            else:
                raise self
        return self._next_response()

    def debug_error_string(self) -> Optional[str]:
        with self._state.condition:
            if self._state.debug_error_string is None:
                raise grpc.experimental.UsageError('Cannot get debug error string until RPC is completed.')
            return _common.decode(self._state.debug_error_string)