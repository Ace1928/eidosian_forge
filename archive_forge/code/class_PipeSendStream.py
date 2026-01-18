from __future__ import annotations
import sys
from typing import TYPE_CHECKING
from . import _core
from ._abc import ReceiveStream, SendStream
from ._core._windows_cffi import _handle, kernel32, raise_winerror
from ._util import ConflictDetector, final
@final
class PipeSendStream(SendStream):
    """Represents a send stream over a Windows named pipe that has been
    opened in OVERLAPPED mode.
    """

    def __init__(self, handle: int) -> None:
        self._handle_holder = _HandleHolder(handle)
        self._conflict_detector = ConflictDetector('another task is currently using this pipe')

    async def send_all(self, data: bytes) -> None:
        with self._conflict_detector:
            if self._handle_holder.closed:
                raise _core.ClosedResourceError('this pipe is already closed')
            if not data:
                await _core.checkpoint()
                return
            try:
                written = await _core.write_overlapped(self._handle_holder.handle, data)
            except BrokenPipeError as ex:
                raise _core.BrokenResourceError from ex
            assert written == len(data)

    async def wait_send_all_might_not_block(self) -> None:
        with self._conflict_detector:
            if self._handle_holder.closed:
                raise _core.ClosedResourceError('This pipe is already closed')
            await _core.checkpoint()

    def close(self) -> None:
        self._handle_holder.close()

    async def aclose(self) -> None:
        self.close()
        await _core.checkpoint()