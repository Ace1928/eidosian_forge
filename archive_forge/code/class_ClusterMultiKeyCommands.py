import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
class ClusterMultiKeyCommands(ClusterCommandsProtocol):
    """
    A class containing commands that handle more than one key
    """

    def _partition_keys_by_slot(self, keys: Iterable[KeyT]) -> Dict[int, List[KeyT]]:
        """Split keys into a dictionary that maps a slot to a list of keys."""
        slots_to_keys = {}
        for key in keys:
            slot = key_slot(self.encoder.encode(key))
            slots_to_keys.setdefault(slot, []).append(key)
        return slots_to_keys

    def _partition_pairs_by_slot(self, mapping: Mapping[AnyKeyT, EncodableT]) -> Dict[int, List[EncodableT]]:
        """Split pairs into a dictionary that maps a slot to a list of pairs."""
        slots_to_pairs = {}
        for pair in mapping.items():
            slot = key_slot(self.encoder.encode(pair[0]))
            slots_to_pairs.setdefault(slot, []).extend(pair)
        return slots_to_pairs

    def _execute_pipeline_by_slot(self, command: str, slots_to_args: Mapping[int, Iterable[EncodableT]]) -> List[Any]:
        read_from_replicas = self.read_from_replicas and command in READ_COMMANDS
        pipe = self.pipeline()
        [pipe.execute_command(command, *slot_args, target_nodes=[self.nodes_manager.get_node_from_slot(slot, read_from_replicas)]) for slot, slot_args in slots_to_args.items()]
        return pipe.execute()

    def _reorder_keys_by_command(self, keys: Iterable[KeyT], slots_to_args: Mapping[int, Iterable[EncodableT]], responses: Iterable[Any]) -> List[Any]:
        results = {k: v for slot_values, response in zip(slots_to_args.values(), responses) for k, v in zip(slot_values, response)}
        return [results[key] for key in keys]

    def mget_nonatomic(self, keys: KeysT, *args: KeyT) -> List[Optional[Any]]:
        """
        Splits the keys into different slots and then calls MGET
        for the keys of every slot. This operation will not be atomic
        if keys belong to more than one slot.

        Returns a list of values ordered identically to ``keys``

        For more information see https://redis.io/commands/mget
        """
        keys = list_or_args(keys, args)
        slots_to_keys = self._partition_keys_by_slot(keys)
        res = self._execute_pipeline_by_slot('MGET', slots_to_keys)
        return self._reorder_keys_by_command(keys, slots_to_keys, res)

    def mset_nonatomic(self, mapping: Mapping[AnyKeyT, EncodableT]) -> List[bool]:
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
        return self._execute_pipeline_by_slot('MSET', slots_to_pairs)

    def _split_command_across_slots(self, command: str, *keys: KeyT) -> int:
        """
        Runs the given command once for the keys
        of each slot. Returns the sum of the return values.
        """
        slots_to_keys = self._partition_keys_by_slot(keys)
        return sum(self._execute_pipeline_by_slot(command, slots_to_keys))

    def exists(self, *keys: KeyT) -> ResponseT:
        """
        Returns the number of ``names`` that exist in the
        whole cluster. The keys are first split up into slots
        and then an EXISTS command is sent for every slot

        For more information see https://redis.io/commands/exists
        """
        return self._split_command_across_slots('EXISTS', *keys)

    def delete(self, *keys: KeyT) -> ResponseT:
        """
        Deletes the given keys in the cluster.
        The keys are first split up into slots
        and then an DEL command is sent for every slot

        Non-existent keys are ignored.
        Returns the number of keys that were deleted.

        For more information see https://redis.io/commands/del
        """
        return self._split_command_across_slots('DEL', *keys)

    def touch(self, *keys: KeyT) -> ResponseT:
        """
        Updates the last access time of given keys across the
        cluster.

        The keys are first split up into slots
        and then an TOUCH command is sent for every slot

        Non-existent keys are ignored.
        Returns the number of keys that were touched.

        For more information see https://redis.io/commands/touch
        """
        return self._split_command_across_slots('TOUCH', *keys)

    def unlink(self, *keys: KeyT) -> ResponseT:
        """
        Remove the specified keys in a different thread.

        The keys are first split up into slots
        and then an TOUCH command is sent for every slot

        Non-existent keys are ignored.
        Returns the number of keys that were unlinked.

        For more information see https://redis.io/commands/unlink
        """
        return self._split_command_across_slots('UNLINK', *keys)