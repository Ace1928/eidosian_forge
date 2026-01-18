from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
@_util.final
class MemoryReceiveStream(ReceiveStream):
    """An in-memory :class:`~trio.abc.ReceiveStream`.

    Args:
      receive_some_hook: An async function, or None. Called from
          :meth:`receive_some`. Can do whatever you like.
      close_hook: A synchronous function, or None. Called from :meth:`close`
          and :meth:`aclose`. Can do whatever you like.

    .. attribute:: receive_some_hook
                   close_hook

       Both hooks are also exposed as attributes on the object, and you can
       change them at any time.

    """

    def __init__(self, receive_some_hook: AsyncHook | None=None, close_hook: SyncHook | None=None):
        self._conflict_detector = _util.ConflictDetector('another task is using this stream')
        self._incoming = _UnboundedByteQueue()
        self._closed = False
        self.receive_some_hook = receive_some_hook
        self.close_hook = close_hook

    async def receive_some(self, max_bytes: int | None=None) -> bytearray:
        """Calls the :attr:`receive_some_hook` (if any), and then retrieves
        data from the internal buffer, blocking if necessary.

        """
        with self._conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            if self._closed:
                raise _core.ClosedResourceError
            if self.receive_some_hook is not None:
                await self.receive_some_hook()
            data = await self._incoming.get(max_bytes)
            if self._closed:
                raise _core.ClosedResourceError
            return data

    def close(self) -> None:
        """Discards any pending data from the internal buffer, and marks this
        stream as closed.

        """
        self._closed = True
        self._incoming.close_and_wipe()
        if self.close_hook is not None:
            self.close_hook()

    async def aclose(self) -> None:
        """Same as :meth:`close`, but async."""
        self.close()
        await _core.checkpoint()

    def put_data(self, data: bytes | bytearray | memoryview) -> None:
        """Appends the given data to the internal buffer."""
        self._incoming.put(data)

    def put_eof(self) -> None:
        """Adds an end-of-file marker to the internal buffer."""
        self._incoming.close()