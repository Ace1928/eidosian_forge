import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
class AsyncClusterMultiKeyCommands(ClusterMultiKeyCommands):
    """
    A class containing commands that handle more than one key
    """

    async def mget_nonatomic(self, keys: KeysT, *args: KeyT) -> List[Optional[Any]]:
        """
        Splits the keys into different slots and then calls MGET
        for the keys of every slot. This operation will not be atomic
        if keys belong to more than one slot.

        Returns a list of values ordered identically to ``keys``

        For more information see https://redis.io/commands/mget
        """
        keys = list_or_args(keys, args)
        slots_to_keys = self._partition_keys_by_slot(keys)
        res = await self._execute_pipeline_by_slot('MGET', slots_to_keys)
        return self._reorder_keys_by_command(keys, slots_to_keys, res)

    async def mset_nonatomic(self, mapping: Mapping[AnyKeyT, EncodableT]) -> List[bool]:
        """
        Sets key/values based on a mapping. Mapping is a dictionary of
        key/value pairs. Both keys and values should be strings or types that
        can be cast to a string via str().

        Splits the keys into different slots and then calls MSET
        for the keys of every slot. This operation will not be atomic
        if keys belong to more than one slot.

        For more information see https://redis.io/commands/mset
        """
        slots_to_pairs = self._partition_pairs_by_slot(mapping)
        return await self._execute_pipeline_by_slot('MSET', slots_to_pairs)

    async def _split_command_across_slots(self, command: str, *keys: KeyT) -> int:
        """
        Runs the given command once for the keys
        of each slot. Returns the sum of the return values.
        """
        slots_to_keys = self._partition_keys_by_slot(keys)
        return sum(await self._execute_pipeline_by_slot(command, slots_to_keys))

    async def _execute_pipeline_by_slot(self, command: str, slots_to_args: Mapping[int, Iterable[EncodableT]]) -> List[Any]:
        if self._initialize:
            await self.initialize()
        read_from_replicas = self.read_from_replicas and command in READ_COMMANDS
        pipe = self.pipeline()
        [pipe.execute_command(command, *slot_args, target_nodes=[self.nodes_manager.get_node_from_slot(slot, read_from_replicas)]) for slot, slot_args in slots_to_args.items()]
        return await pipe.execute()