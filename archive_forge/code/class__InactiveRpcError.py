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
class _InactiveRpcError(grpc.RpcError, grpc.Call, grpc.Future):
    """An RPC error not tied to the execution of a particular RPC.

    The RPC represented by the state object must not be in-progress or
    cancelled.

    Attributes:
      _state: An instance of _RPCState.
    """
    _state: _RPCState

    def __init__(self, state: _RPCState):
        with state.condition:
            self._state = _RPCState((), copy.deepcopy(state.initial_metadata), copy.deepcopy(state.trailing_metadata), state.code, copy.deepcopy(state.details))
            self._state.response = copy.copy(state.response)
            self._state.debug_error_string = copy.copy(state.debug_error_string)

    def initial_metadata(self) -> Optional[MetadataType]:
        return self._state.initial_metadata

    def trailing_metadata(self) -> Optional[MetadataType]:
        return self._state.trailing_metadata

    def code(self) -> Optional[grpc.StatusCode]:
        return self._state.code

    def details(self) -> Optional[str]:
        return _common.decode(self._state.details)

    def debug_error_string(self) -> Optional[str]:
        return _common.decode(self._state.debug_error_string)

    def _repr(self) -> str:
        return _rpc_state_string(self.__class__.__name__, self._state)

    def __repr__(self) -> str:
        return self._repr()

    def __str__(self) -> str:
        return self._repr()

    def cancel(self) -> bool:
        """See grpc.Future.cancel."""
        return False

    def cancelled(self) -> bool:
        """See grpc.Future.cancelled."""
        return False

    def running(self) -> bool:
        """See grpc.Future.running."""
        return False

    def done(self) -> bool:
        """See grpc.Future.done."""
        return True

    def result(self, timeout: Optional[float]=None) -> Any:
        """See grpc.Future.result."""
        raise self

    def exception(self, timeout: Optional[float]=None) -> Optional[Exception]:
        """See grpc.Future.exception."""
        return self

    def traceback(self, timeout: Optional[float]=None) -> Optional[types.TracebackType]:
        """See grpc.Future.traceback."""
        try:
            raise self
        except grpc.RpcError:
            return sys.exc_info()[2]

    def add_done_callback(self, fn: Callable[[grpc.Future], None], timeout: Optional[float]=None) -> None:
        """See grpc.Future.add_done_callback."""
        fn(self)