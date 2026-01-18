from functools import partial
from ..base import AsyncBase
from ..threadpool.utils import (
@delegate_to_executor('cleanup')
@proxy_property_directly('name')
class AsyncTemporaryDirectory:
    """Async wrapper for TemporaryDirectory class"""

    def __init__(self, file, loop, executor):
        self._file = file
        self._loop = loop
        self._executor = executor

    async def close(self):
        await self.cleanup()