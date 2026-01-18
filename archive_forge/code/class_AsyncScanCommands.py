import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class AsyncScanCommands(ScanCommands):

    async def scan_iter(self, match: Union[PatternT, None]=None, count: Union[int, None]=None, _type: Union[str, None]=None, **kwargs) -> AsyncIterator:
        """
        Make an iterator using the SCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` provides a hint to Redis about the number of keys to
            return per batch.

        ``_type`` filters the returned values by a particular Redis type.
            Stock Redis instances allow for the following types:
            HASH, LIST, SET, STREAM, STRING, ZSET
            Additionally, Redis modules can expose other types as well.
        """
        cursor = '0'
        while cursor != 0:
            cursor, data = await self.scan(cursor=cursor, match=match, count=count, _type=_type, **kwargs)
            for d in data:
                yield d

    async def sscan_iter(self, name: KeyT, match: Union[PatternT, None]=None, count: Union[int, None]=None) -> AsyncIterator:
        """
        Make an iterator using the SSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        cursor = '0'
        while cursor != 0:
            cursor, data = await self.sscan(name, cursor=cursor, match=match, count=count)
            for d in data:
                yield d

    async def hscan_iter(self, name: str, match: Union[PatternT, None]=None, count: Union[int, None]=None) -> AsyncIterator:
        """
        Make an iterator using the HSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns
        """
        cursor = '0'
        while cursor != 0:
            cursor, data = await self.hscan(name, cursor=cursor, match=match, count=count)
            for it in data.items():
                yield it

    async def zscan_iter(self, name: KeyT, match: Union[PatternT, None]=None, count: Union[int, None]=None, score_cast_func: Union[type, Callable]=float) -> AsyncIterator:
        """
        Make an iterator using the ZSCAN command so that the client doesn't
        need to remember the cursor position.

        ``match`` allows for filtering the keys by pattern

        ``count`` allows for hint the minimum number of returns

        ``score_cast_func`` a callable used to cast the score return value
        """
        cursor = '0'
        while cursor != 0:
            cursor, data = await self.zscan(name, cursor=cursor, match=match, count=count, score_cast_func=score_cast_func)
            for d in data:
                yield d