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
class ClientCallDetails(collections.namedtuple('ClientCallDetails', ('method', 'timeout', 'metadata', 'credentials', 'wait_for_ready')), grpc.ClientCallDetails):
    """Describes an RPC to be invoked.

    This is an EXPERIMENTAL API.

    Args:
        method: The method name of the RPC.
        timeout: An optional duration of time in seconds to allow for the RPC.
        metadata: Optional metadata to be transmitted to the service-side of
          the RPC.
        credentials: An optional CallCredentials for the RPC.
        wait_for_ready: An optional flag to enable :term:`wait_for_ready` mechanism.
    """
    method: str
    timeout: Optional[float]
    metadata: Optional[Metadata]
    credentials: Optional[grpc.CallCredentials]
    wait_for_ready: Optional[bool]