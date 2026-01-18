import asyncio
import collections
import random
import socket
import warnings
from typing import (
from aiokeydb.v1.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from aiokeydb.v1.asyncio.core import ResponseCallbackT
from aiokeydb.v1.asyncio.connection import (
from aiokeydb.v1.asyncio.parser import CommandsParser
from aiokeydb.v1.core import EMPTY_RESPONSE, NEVER_DECODE, AbstractKeyDB
from aiokeydb.v1.cluster import (
from aiokeydb.v1.commands import READ_COMMANDS, AsyncKeyDBClusterCommands
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.typing import AnyKeyT, EncodableT, KeyT
from aiokeydb.v1.utils import dict_merge, safe_str, str_if_bytes
class AsyncNodesManager:
    __slots__ = ('_moved_exception', 'connection_kwargs', 'default_node', 'nodes_cache', 'read_load_balancer', 'require_full_coverage', 'slots_cache', 'startup_nodes')

    def __init__(self, startup_nodes: List['AsyncClusterNode'], require_full_coverage: bool, connection_kwargs: Dict[str, Any]) -> None:
        self.startup_nodes = {node.name: node for node in startup_nodes}
        self.require_full_coverage = require_full_coverage
        self.connection_kwargs = connection_kwargs
        self.default_node: 'AsyncClusterNode' = None
        self.nodes_cache: Dict[str, 'AsyncClusterNode'] = {}
        self.slots_cache: Dict[int, List['AsyncClusterNode']] = {}
        self.read_load_balancer = LoadBalancer()
        self._moved_exception: MovedError = None

    def get_node(self, host: Optional[str]=None, port: Optional[int]=None, node_name: Optional[str]=None) -> Optional['AsyncClusterNode']:
        if host and port:
            if host == 'localhost':
                host = socket.gethostbyname(host)
            return self.nodes_cache.get(get_node_name(host=host, port=port))
        elif node_name:
            return self.nodes_cache.get(node_name)
        else:
            raise DataError('get_node requires one of the following: 1. node name 2. host and port')

    def set_nodes(self, old: Dict[str, 'AsyncClusterNode'], new: Dict[str, 'AsyncClusterNode'], remove_old: bool=False) -> None:
        if remove_old:
            for name in list(old.keys()):
                if name not in new:
                    asyncio.ensure_future(old.pop(name).disconnect())
        for name, node in new.items():
            if name in old:
                if old[name] is node:
                    continue
                asyncio.ensure_future(old[name].disconnect())
            old[name] = node

    def _update_moved_slots(self) -> None:
        e = self._moved_exception
        redirected_node = self.get_node(host=e.host, port=e.port)
        if redirected_node:
            if redirected_node.server_type != PRIMARY:
                redirected_node.server_type = PRIMARY
        else:
            redirected_node = AsyncClusterNode(e.host, e.port, PRIMARY, **self.connection_kwargs)
            self.set_nodes(self.nodes_cache, {redirected_node.name: redirected_node})
        if redirected_node in self.slots_cache[e.slot_id]:
            old_primary = self.slots_cache[e.slot_id][0]
            old_primary.server_type = REPLICA
            self.slots_cache[e.slot_id].append(old_primary)
            self.slots_cache[e.slot_id].remove(redirected_node)
            self.slots_cache[e.slot_id][0] = redirected_node
            if self.default_node == old_primary:
                self.default_node = redirected_node
        else:
            self.slots_cache[e.slot_id] = [redirected_node]
        self._moved_exception = None

    def get_node_from_slot(self, slot: int, read_from_replicas: bool=False) -> 'AsyncClusterNode':
        if self._moved_exception:
            self._update_moved_slots()
        try:
            if read_from_replicas:
                primary_name = self.slots_cache[slot][0].name
                node_idx = self.read_load_balancer.get_server_index(primary_name, len(self.slots_cache[slot]))
                return self.slots_cache[slot][node_idx]
            return self.slots_cache[slot][0]
        except (IndexError, TypeError):
            raise SlotNotCoveredError(f'Slot "{slot}" not covered by the cluster. "require_full_coverage={self.require_full_coverage}"')

    def get_nodes_by_server_type(self, server_type: str) -> List['AsyncClusterNode']:
        return [node for node in self.nodes_cache.values() if node.server_type == server_type]

    async def initialize(self) -> None:
        self.read_load_balancer.reset()
        tmp_nodes_cache: Dict[str, 'AsyncClusterNode'] = {}
        tmp_slots: Dict[int, List['AsyncClusterNode']] = {}
        disagreements = []
        startup_nodes_reachable = False
        fully_covered = False
        exception = None
        for startup_node in self.startup_nodes.values():
            try:
                if not (await startup_node.execute_command('INFO')).get('cluster_enabled'):
                    raise KeyDBClusterException('Cluster mode is not enabled on this node')
                cluster_slots = await startup_node.execute_command('CLUSTER SLOTS')
                startup_nodes_reachable = True
            except (ConnectionError, TimeoutError) as e:
                exception = e
                continue
            except ResponseError as e:
                message = e.__str__()
                if 'CLUSTERDOWN' in message or 'MASTERDOWN' in message:
                    continue
                else:
                    raise KeyDBClusterException(f'ERROR sending "cluster slots" command to redis server: {startup_node}. error: {message}')
            except Exception as e:
                message = e.__str__()
                raise KeyDBClusterException(f'ERROR sending "cluster slots" command to redis server {startup_node.name}. error: {message}')
            if len(cluster_slots) == 1 and (not cluster_slots[0][2][0]) and (len(self.startup_nodes) == 1):
                cluster_slots[0][2][0] = startup_node.host
            for slot in cluster_slots:
                for i in range(2, len(slot)):
                    slot[i] = [str_if_bytes(val) for val in slot[i]]
                primary_node = slot[2]
                host = primary_node[0]
                if host == '':
                    host = startup_node.host
                port = int(primary_node[1])
                target_node = tmp_nodes_cache.get(get_node_name(host, port))
                if not target_node:
                    target_node = AsyncClusterNode(host, port, PRIMARY, **self.connection_kwargs)
                tmp_nodes_cache[target_node.name] = target_node
                for i in range(int(slot[0]), int(slot[1]) + 1):
                    if i not in tmp_slots:
                        tmp_slots[i] = []
                        tmp_slots[i].append(target_node)
                        replica_nodes = [slot[j] for j in range(3, len(slot))]
                        for replica_node in replica_nodes:
                            host = replica_node[0]
                            port = replica_node[1]
                            target_replica_node = tmp_nodes_cache.get(get_node_name(host, port))
                            if not target_replica_node:
                                target_replica_node = AsyncClusterNode(host, port, REPLICA, **self.connection_kwargs)
                            tmp_slots[i].append(target_replica_node)
                            tmp_nodes_cache[target_replica_node.name] = target_replica_node
                    else:
                        tmp_slot = tmp_slots[i][0]
                        if tmp_slot.name != target_node.name:
                            disagreements.append(f'{tmp_slot.name} vs {target_node.name} on slot: {i}')
                            if len(disagreements) > 5:
                                raise KeyDBClusterException(f'startup_nodes could not agree on a valid slots cache: {', '.join(disagreements)}')
            fully_covered = True
            for i in range(REDIS_CLUSTER_HASH_SLOTS):
                if i not in tmp_slots:
                    fully_covered = False
                    break
            if fully_covered:
                break
        if not startup_nodes_reachable:
            raise KeyDBClusterException('KeyDB Cluster cannot be connected. Please provide at least one reachable node. ') from exception
        if not fully_covered and self.require_full_coverage:
            raise KeyDBClusterException(f'All slots are not covered after query all startup_nodes. {len(tmp_slots)} of {REDIS_CLUSTER_HASH_SLOTS} covered...')
        self.slots_cache = tmp_slots
        self.set_nodes(self.nodes_cache, tmp_nodes_cache, remove_old=True)
        self.set_nodes(self.startup_nodes, self.nodes_cache, remove_old=True)
        self.default_node = self.get_nodes_by_server_type(PRIMARY)[0]
        self._moved_exception = None

    async def close(self, attr: str='nodes_cache') -> None:
        self.default_node = None
        await asyncio.gather(*(asyncio.ensure_future(node.disconnect()) for node in getattr(self, attr).values()))