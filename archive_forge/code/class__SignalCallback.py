from types import SimpleNamespace
from typing import TYPE_CHECKING, Awaitable, Optional, Protocol, Type, TypeVar
import attr
from aiosignal import Signal
from multidict import CIMultiDict
from yarl import URL
from .client_reqrep import ClientResponse
class _SignalCallback(Protocol[_ParamT_contra]):

    def __call__(self, __client_session: ClientSession, __trace_config_ctx: SimpleNamespace, __params: _ParamT_contra) -> Awaitable[None]:
        ...