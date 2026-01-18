import asyncio
import functools
from typing import Tuple
class LruCacheDecoratedMethod(object):

    @functools.lru_cache()
    def lru_cache_in_class(self, arg1):
        return arg1