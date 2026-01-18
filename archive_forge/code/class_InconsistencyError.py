from __future__ import annotations
from socket import timeout as TimeoutError
from types import TracebackType
from typing import TYPE_CHECKING, TypeVar
from amqp import ChannelError, ConnectionError, ResourceError
class InconsistencyError(ConnectionError):
    """Data or environment has been found to be inconsistent.

    Depending on the cause it may be possible to retry the operation.
    """