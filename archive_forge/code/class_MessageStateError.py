from __future__ import annotations
from socket import timeout as TimeoutError
from types import TracebackType
from typing import TYPE_CHECKING, TypeVar
from amqp import ChannelError, ConnectionError, ResourceError
class MessageStateError(KombuError):
    """The message has already been acknowledged."""