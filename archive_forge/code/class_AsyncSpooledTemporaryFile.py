from functools import partial
from ..base import AsyncBase
from ..threadpool.utils import (
@delegate_to_executor('fileno', 'rollover')
@cond_delegate_to_executor('close', 'flush', 'isatty', 'read', 'readline', 'readlines', 'seek', 'tell', 'truncate')
@proxy_property_directly('closed', 'encoding', 'mode', 'name', 'newlines')
class AsyncSpooledTemporaryFile(AsyncBase):
    """Async wrapper for SpooledTemporaryFile class"""

    async def _check(self):
        if self._file._rolled:
            return
        max_size = self._file._max_size
        if max_size and self._file.tell() > max_size:
            await self.rollover()

    async def write(self, s):
        """Implementation to anticipate rollover"""
        if self._file._rolled:
            cb = partial(self._file.write, s)
            return await self._loop.run_in_executor(self._executor, cb)
        else:
            file = self._file._file
            rv = file.write(s)
            await self._check()
            return rv

    async def writelines(self, iterable):
        """Implementation to anticipate rollover"""
        if self._file._rolled:
            cb = partial(self._file.writelines, iterable)
            return await self._loop.run_in_executor(self._executor, cb)
        else:
            file = self._file._file
            rv = file.writelines(iterable)
            await self._check()
            return rv