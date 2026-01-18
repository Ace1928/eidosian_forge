from __future__ import annotations
import contextlib
import operator as _operator
import ssl as _stdlib_ssl
from enum import Enum as _Enum
from typing import TYPE_CHECKING, Any, ClassVar, Final as TFinal, Generic, TypeVar
import trio
from . import _sync
from ._highlevel_generic import aclose_forcefully
from ._util import ConflictDetector, final
from .abc import Listener, Stream
@final
class SSLListener(Listener[SSLStream[T_Stream]]):
    """A :class:`~trio.abc.Listener` for SSL/TLS-encrypted servers.

    :class:`SSLListener` wraps around another Listener, and converts
    all incoming connections to encrypted connections by wrapping them
    in a :class:`SSLStream`.

    Args:
      transport_listener (~trio.abc.Listener): The listener whose incoming
          connections will be wrapped in :class:`SSLStream`.

      ssl_context (~ssl.SSLContext): The :class:`~ssl.SSLContext` that will be
          used for incoming connections.

      https_compatible (bool): Passed on to :class:`SSLStream`.

    Attributes:
      transport_listener (trio.abc.Listener): The underlying listener that was
          passed to ``__init__``.

    """

    def __init__(self, transport_listener: Listener[T_Stream], ssl_context: _stdlib_ssl.SSLContext, *, https_compatible: bool=False) -> None:
        self.transport_listener = transport_listener
        self._ssl_context = ssl_context
        self._https_compatible = https_compatible

    async def accept(self) -> SSLStream[T_Stream]:
        """Accept the next connection and wrap it in an :class:`SSLStream`.

        See :meth:`trio.abc.Listener.accept` for details.

        """
        transport_stream = await self.transport_listener.accept()
        return SSLStream(transport_stream, self._ssl_context, server_side=True, https_compatible=self._https_compatible)

    async def aclose(self) -> None:
        """Close the transport listener."""
        await self.transport_listener.aclose()