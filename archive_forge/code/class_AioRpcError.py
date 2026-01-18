import asyncio
import enum
from functools import partial
import inspect
import logging
import traceback
from typing import Any, AsyncIterator, Generator, Generic, Optional, Tuple
import grpc
from grpc import _common
from grpc._cython import cygrpc
from . import _base_call
from ._metadata import Metadata
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import MetadatumType
from ._typing import RequestIterableType
from ._typing import RequestType
from ._typing import ResponseType
from ._typing import SerializingFunction
class AioRpcError(grpc.RpcError):
    """An implementation of RpcError to be used by the asynchronous API.

    Raised RpcError is a snapshot of the final status of the RPC, values are
    determined. Hence, its methods no longer needs to be coroutines.
    """
    _code: grpc.StatusCode
    _details: Optional[str]
    _initial_metadata: Optional[Metadata]
    _trailing_metadata: Optional[Metadata]
    _debug_error_string: Optional[str]

    def __init__(self, code: grpc.StatusCode, initial_metadata: Metadata, trailing_metadata: Metadata, details: Optional[str]=None, debug_error_string: Optional[str]=None) -> None:
        """Constructor.

        Args:
          code: The status code with which the RPC has been finalized.
          details: Optional details explaining the reason of the error.
          initial_metadata: Optional initial metadata that could be sent by the
            Server.
          trailing_metadata: Optional metadata that could be sent by the Server.
        """
        super().__init__()
        self._code = code
        self._details = details
        self._initial_metadata = initial_metadata
        self._trailing_metadata = trailing_metadata
        self._debug_error_string = debug_error_string

    def code(self) -> grpc.StatusCode:
        """Accesses the status code sent by the server.

        Returns:
          The `grpc.StatusCode` status code.
        """
        return self._code

    def details(self) -> Optional[str]:
        """Accesses the details sent by the server.

        Returns:
          The description of the error.
        """
        return self._details

    def initial_metadata(self) -> Metadata:
        """Accesses the initial metadata sent by the server.

        Returns:
          The initial metadata received.
        """
        return self._initial_metadata

    def trailing_metadata(self) -> Metadata:
        """Accesses the trailing metadata sent by the server.

        Returns:
          The trailing metadata received.
        """
        return self._trailing_metadata

    def debug_error_string(self) -> str:
        """Accesses the debug error string sent by the server.

        Returns:
          The debug error string received.
        """
        return self._debug_error_string

    def _repr(self) -> str:
        """Assembles the error string for the RPC error."""
        return _NON_OK_CALL_REPRESENTATION.format(self.__class__.__name__, self._code, self._details, self._debug_error_string)

    def __repr__(self) -> str:
        return self._repr()

    def __str__(self) -> str:
        return self._repr()

    def __reduce__(self):
        return (type(self), (self._code, self._initial_metadata, self._trailing_metadata, self._details, self._debug_error_string))