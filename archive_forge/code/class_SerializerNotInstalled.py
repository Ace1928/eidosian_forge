from __future__ import annotations
from socket import timeout as TimeoutError
from types import TracebackType
from typing import TYPE_CHECKING, TypeVar
from amqp import ChannelError, ConnectionError, ResourceError
class SerializerNotInstalled(KombuError):
    """Support for the requested serialization type is not installed."""