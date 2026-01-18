import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from ray._raylet import GcsClient
from ray.core.generated import (
import ray._private.utils
from ray._private.ray_constants import env_integer
class AsyncProxy:

    def __init__(self, inner, loop, executor):
        self.inner = inner
        self.loop = loop
        self.executor = executor

    def _function_to_async(self, func):

        async def wrapper(*args, **kwargs):
            return await self.loop.run_in_executor(self.executor, func, *args, **kwargs)
        return wrapper

    def __getattr__(self, name):
        """
        If attr is callable, wrap it into an async function.
        """
        attr = getattr(self.inner, name)
        if callable(attr):
            return self._function_to_async(attr)
        else:
            return attr