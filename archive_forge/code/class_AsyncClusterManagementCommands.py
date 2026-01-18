import asyncio
from typing import (
from redis.compat import Literal
from redis.crc import key_slot
from redis.exceptions import RedisClusterException, RedisError
from redis.typing import (
from .core import (
from .helpers import list_or_args
from .redismodules import AsyncRedisModuleCommands, RedisModuleCommands
class AsyncClusterManagementCommands(ClusterManagementCommands, AsyncManagementCommands):
    """
    A class for Redis Cluster management commands

    The class inherits from Redis's core ManagementCommands class and do the
    required adjustments to work with cluster mode
    """

    async def cluster_delslots(self, *slots: EncodableT) -> List[bool]:
        """
        Set hash slots as unbound in the cluster.
        It determines by it self what node the slot is in and sends it there

        Returns a list of the results for each processed slot.

        For more information see https://redis.io/commands/cluster-delslots
        """
        return await asyncio.gather(*(asyncio.create_task(self.execute_command('CLUSTER DELSLOTS', slot)) for slot in slots))