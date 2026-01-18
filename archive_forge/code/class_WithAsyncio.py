import asyncio
import functools
from typing import Tuple
class WithAsyncio(object):

    async def double(self, count=0):
        return 2 * count