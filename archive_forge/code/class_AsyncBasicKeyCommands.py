import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
class AsyncBasicKeyCommands(BasicKeyCommands):

    def __delitem__(self, name: KeyT):
        raise TypeError('Async Redis client does not support class deletion')

    def __contains__(self, name: KeyT):
        raise TypeError('Async Redis client does not support class inclusion')

    def __getitem__(self, name: KeyT):
        raise TypeError('Async Redis client does not support class retrieval')

    def __setitem__(self, name: KeyT, value: EncodableT):
        raise TypeError('Async Redis client does not support class assignment')

    async def watch(self, *names: KeyT) -> None:
        return super().watch(*names)

    async def unwatch(self) -> None:
        return super().unwatch()