import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
def _reorder_keys_by_command(self, keys: Iterable[KeyT], slots_to_args: Mapping[int, Iterable[EncodableT]], responses: Iterable[Any]) -> List[Any]:
    results = {k: v for slot_values, response in zip(slots_to_args.values(), responses) for k, v in zip(slot_values, response)}
    return [results[key] for key in keys]