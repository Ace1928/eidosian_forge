from __future__ import annotations
import operator
from typing import TYPE_CHECKING, Awaitable, Callable, TypeVar
from .. import _core, _util
from .._highlevel_generic import StapledStream
from ..abc import ReceiveStream, SendStream
@_util.final
class MemorySendStream(SendStream):
    """An in-memory :class:`~trio.abc.SendStream`.

    Args:
      send_all_hook: An async function, or None. Called from
          :meth:`send_all`. Can do whatever you like.
      wait_send_all_might_not_block_hook: An async function, or None. Called
          from :meth:`wait_send_all_might_not_block`. Can do whatever you
          like.
      close_hook: A synchronous function, or None. Called from :meth:`close`
          and :meth:`aclose`. Can do whatever you like.

    .. attribute:: send_all_hook
                   wait_send_all_might_not_block_hook
                   close_hook

       All of these hooks are also exposed as attributes on the object, and
       you can change them at any time.

    """

    def __init__(self, send_all_hook: AsyncHook | None=None, wait_send_all_might_not_block_hook: AsyncHook | None=None, close_hook: SyncHook | None=None):
        self._conflict_detector = _util.ConflictDetector('another task is using this stream')
        self._outgoing = _UnboundedByteQueue()
        self.send_all_hook = send_all_hook
        self.wait_send_all_might_not_block_hook = wait_send_all_might_not_block_hook
        self.close_hook = close_hook

    async def send_all(self, data: bytes | bytearray | memoryview) -> None:
        """Places the given data into the object's internal buffer, and then
        calls the :attr:`send_all_hook` (if any).

        """
        with self._conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            self._outgoing.put(data)
            if self.send_all_hook is not None:
                await self.send_all_hook()

    async def wait_send_all_might_not_block(self) -> None:
        """Calls the :attr:`wait_send_all_might_not_block_hook` (if any), and
        then returns immediately.

        """
        with self._conflict_detector:
            await _core.checkpoint()
            await _core.checkpoint()
            self._outgoing.put(b'')
            if self.wait_send_all_might_not_block_hook is not None:
                await self.wait_send_all_might_not_block_hook()

    def close(self) -> None:
        """Marks this stream as closed, and then calls the :attr:`close_hook`
        (if any).

        """
        self._outgoing.close()
        if self.close_hook is not None:
            self.close_hook()

    async def aclose(self) -> None:
        """Same as :meth:`close`, but async."""
        self.close()
        await _core.checkpoint()

    async def get_data(self, max_bytes: int | None=None) -> bytearray:
        """Retrieves data from the internal buffer, blocking if necessary.

        Args:
          max_bytes (int or None): The maximum amount of data to
              retrieve. None (the default) means to retrieve all the data
              that's present (but still blocks until at least one byte is
              available).

        Returns:
          If this stream has been closed, an empty bytearray. Otherwise, the
          requested data.

        """
        return await self._outgoing.get(max_bytes)

    def get_data_nowait(self, max_bytes: int | None=None) -> bytearray:
        """Retrieves data from the internal buffer, but doesn't block.

        See :meth:`get_data` for details.

        Raises:
          trio.WouldBlock: if no data is available to retrieve.

        """
        return self._outgoing.get_nowait(max_bytes)