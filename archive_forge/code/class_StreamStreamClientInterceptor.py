from abc import ABCMeta
from abc import abstractmethod
import asyncio
import collections
import functools
from typing import (
import grpc
from grpc._cython import cygrpc
from . import _base_call
from ._call import AioRpcError
from ._call import StreamStreamCall
from ._call import StreamUnaryCall
from ._call import UnaryStreamCall
from ._call import UnaryUnaryCall
from ._call import _API_STYLE_ERROR
from ._call import _RPC_ALREADY_FINISHED_DETAILS
from ._call import _RPC_HALF_CLOSED_DETAILS
from ._metadata import Metadata
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import RequestIterableType
from ._typing import RequestType
from ._typing import ResponseIterableType
from ._typing import ResponseType
from ._typing import SerializingFunction
from ._utils import _timeout_to_deadline
class StreamStreamClientInterceptor(ClientInterceptor, metaclass=ABCMeta):
    """Affords intercepting stream-stream invocations."""

    @abstractmethod
    async def intercept_stream_stream(self, continuation: Callable[[ClientCallDetails, RequestType], StreamStreamCall], client_call_details: ClientCallDetails, request_iterator: RequestIterableType) -> Union[ResponseIterableType, StreamStreamCall]:
        """Intercepts a stream-stream invocation asynchronously.

        Within the interceptor the usage of the call methods like `write` or
        even awaiting the call should be done carefully, since the caller
        could be expecting an untouched call, for example for start writing
        messages to it.

        The function could return the call object or an asynchronous
        iterator, in case of being an asyncrhonous iterator this will
        become the source of the reads done by the caller.

        Args:
          continuation: A coroutine that proceeds with the invocation by
            executing the next interceptor in the chain or invoking the
            actual RPC on the underlying Channel. It is the interceptor's
            responsibility to call it if it decides to move the RPC forward.
            The interceptor can use
            `call = await continuation(client_call_details, request_iterator)`
            to continue with the RPC. `continuation` returns the call to the
            RPC.
          client_call_details: A ClientCallDetails object describing the
            outgoing RPC.
          request_iterator: The request iterator that will produce requests
            for the RPC.

        Returns:
          The RPC Call or an asynchronous iterator.

        Raises:
          AioRpcError: Indicating that the RPC terminated with non-OK status.
          asyncio.CancelledError: Indicating that the RPC was canceled.
        """