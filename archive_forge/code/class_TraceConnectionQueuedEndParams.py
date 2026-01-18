from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, Optional, Protocol, Type, TypeVar
import attr
from aiosignal import Signal
from multidict import CIMultiDict
from yarl import URL
from .client_reqrep import ClientResponse
@attr.s(auto_attribs=True, frozen=True, slots=True)
class TraceConnectionQueuedEndParams:
    """Parameters sent by the `on_connection_queued_end` signal"""