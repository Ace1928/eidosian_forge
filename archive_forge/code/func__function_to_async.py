import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from ray._raylet import GcsClient
from ray.core.generated import (
import ray._private.utils
from ray._private.ray_constants import env_integer
def _function_to_async(self, func):

    async def wrapper(*args, **kwargs):
        return await self.loop.run_in_executor(self.executor, func, *args, **kwargs)
    return wrapper