from types import coroutine
from collections.abc import Coroutine
from asyncio import get_running_loop
class AiofilesContextManager(_ContextManager):
    """An adjusted async context manager for aiofiles."""

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await get_running_loop().run_in_executor(None, self._obj._file.__exit__, exc_type, exc_val, exc_tb)
        self._obj = None