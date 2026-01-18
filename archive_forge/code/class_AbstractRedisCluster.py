import random
import socket
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from redis._parsers import CommandsParser, Encoder
from redis._parsers.helpers import parse_scan
from redis.backoff import default_backoff
from redis.client import CaseInsensitiveDict, PubSub, Redis
from redis.commands import READ_COMMANDS, RedisClusterCommands
from redis.commands.helpers import list_or_args
from redis.connection import ConnectionPool, DefaultParser, parse_url
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
class AbstractRedisCluster:
    RedisClusterRequestTTL = 16
    PRIMARIES = 'primaries'
    REPLICAS = 'replicas'
    ALL_NODES = 'all'
    RANDOM = 'random'
    DEFAULT_NODE = 'default-node'
    NODE_FLAGS = {PRIMARIES, REPLICAS, ALL_NODES, RANDOM, DEFAULT_NODE}
    COMMAND_FLAGS = dict_merge(list_keys_to_dict(['ACL CAT', 'ACL DELUSER', 'ACL DRYRUN', 'ACL GENPASS', 'ACL GETUSER', 'ACL HELP', 'ACL LIST', 'ACL LOG', 'ACL LOAD', 'ACL SAVE', 'ACL SETUSER', 'ACL USERS', 'ACL WHOAMI', 'AUTH', 'CLIENT LIST', 'CLIENT SETINFO', 'CLIENT SETNAME', 'CLIENT GETNAME', 'CONFIG SET', 'CONFIG REWRITE', 'CONFIG RESETSTAT', 'TIME', 'PUBSUB CHANNELS', 'PUBSUB NUMPAT', 'PUBSUB NUMSUB', 'PUBSUB SHARDCHANNELS', 'PUBSUB SHARDNUMSUB', 'PING', 'INFO', 'SHUTDOWN', 'KEYS', 'DBSIZE', 'BGSAVE', 'SLOWLOG GET', 'SLOWLOG LEN', 'SLOWLOG RESET', 'WAIT', 'WAITAOF', 'SAVE', 'MEMORY PURGE', 'MEMORY MALLOC-STATS', 'MEMORY STATS', 'LASTSAVE', 'CLIENT TRACKINGINFO', 'CLIENT PAUSE', 'CLIENT UNPAUSE', 'CLIENT UNBLOCK', 'CLIENT ID', 'CLIENT REPLY', 'CLIENT GETREDIR', 'CLIENT INFO', 'CLIENT KILL', 'READONLY', 'CLUSTER INFO', 'CLUSTER MEET', 'CLUSTER MYSHARDID', 'CLUSTER NODES', 'CLUSTER REPLICAS', 'CLUSTER RESET', 'CLUSTER SET-CONFIG-EPOCH', 'CLUSTER SLOTS', 'CLUSTER SHARDS', 'CLUSTER COUNT-FAILURE-REPORTS', 'CLUSTER KEYSLOT', 'COMMAND', 'COMMAND COUNT', 'COMMAND LIST', 'COMMAND GETKEYS', 'CONFIG GET', 'DEBUG', 'RANDOMKEY', 'READONLY', 'READWRITE', 'TIME', 'TFUNCTION LOAD', 'TFUNCTION DELETE', 'TFUNCTION LIST', 'TFCALL', 'TFCALLASYNC', 'GRAPH.CONFIG', 'LATENCY HISTORY', 'LATENCY LATEST', 'LATENCY RESET', 'MODULE LIST', 'MODULE LOAD', 'MODULE UNLOAD', 'MODULE LOADEX'], DEFAULT_NODE), list_keys_to_dict(['FLUSHALL', 'FLUSHDB', 'FUNCTION DELETE', 'FUNCTION FLUSH', 'FUNCTION LIST', 'FUNCTION LOAD', 'FUNCTION RESTORE', 'REDISGEARS_2.REFRESHCLUSTER', 'SCAN', 'SCRIPT EXISTS', 'SCRIPT FLUSH', 'SCRIPT LOAD'], PRIMARIES), list_keys_to_dict(['FUNCTION DUMP'], RANDOM), list_keys_to_dict(['CLUSTER COUNTKEYSINSLOT', 'CLUSTER DELSLOTS', 'CLUSTER DELSLOTSRANGE', 'CLUSTER GETKEYSINSLOT', 'CLUSTER SETSLOT'], SLOT_ID))
    SEARCH_COMMANDS = (['FT.CREATE', 'FT.SEARCH', 'FT.AGGREGATE', 'FT.EXPLAIN', 'FT.EXPLAINCLI', 'FT,PROFILE', 'FT.ALTER', 'FT.DROPINDEX', 'FT.ALIASADD', 'FT.ALIASUPDATE', 'FT.ALIASDEL', 'FT.TAGVALS', 'FT.SUGADD', 'FT.SUGGET', 'FT.SUGDEL', 'FT.SUGLEN', 'FT.SYNUPDATE', 'FT.SYNDUMP', 'FT.SPELLCHECK', 'FT.DICTADD', 'FT.DICTDEL', 'FT.DICTDUMP', 'FT.INFO', 'FT._LIST', 'FT.CONFIG', 'FT.ADD', 'FT.DEL', 'FT.DROP', 'FT.GET', 'FT.MGET', 'FT.SYNADD'],)
    CLUSTER_COMMANDS_RESPONSE_CALLBACKS = {'CLUSTER SLOTS': parse_cluster_slots, 'CLUSTER SHARDS': parse_cluster_shards, 'CLUSTER MYSHARDID': parse_cluster_myshardid}
    RESULT_CALLBACKS = dict_merge(list_keys_to_dict(['PUBSUB NUMSUB', 'PUBSUB SHARDNUMSUB'], parse_pubsub_numsub), list_keys_to_dict(['PUBSUB NUMPAT'], lambda command, res: sum(list(res.values()))), list_keys_to_dict(['KEYS', 'PUBSUB CHANNELS', 'PUBSUB SHARDCHANNELS'], merge_result), list_keys_to_dict(['PING', 'CONFIG SET', 'CONFIG REWRITE', 'CONFIG RESETSTAT', 'CLIENT SETNAME', 'BGSAVE', 'SLOWLOG RESET', 'SAVE', 'MEMORY PURGE', 'CLIENT PAUSE', 'CLIENT UNPAUSE'], lambda command, res: all(res.values()) if isinstance(res, dict) else res), list_keys_to_dict(['DBSIZE', 'WAIT'], lambda command, res: sum(res.values()) if isinstance(res, dict) else res), list_keys_to_dict(['CLIENT UNBLOCK'], lambda command, res: 1 if sum(res.values()) > 0 else 0), list_keys_to_dict(['SCAN'], parse_scan_result), list_keys_to_dict(['SCRIPT LOAD'], lambda command, res: list(res.values()).pop()), list_keys_to_dict(['SCRIPT EXISTS'], lambda command, res: [all(k) for k in zip(*res.values())]), list_keys_to_dict(['SCRIPT FLUSH'], lambda command, res: all(res.values())))
    ERRORS_ALLOW_RETRY = (ConnectionError, TimeoutError, ClusterDownError)

    def replace_default_node(self, target_node: 'ClusterNode'=None) -> None:
        """Replace the default cluster node.
        A random cluster node will be chosen if target_node isn't passed, and primaries
        will be prioritized. The default node will not be changed if there are no other
        nodes in the cluster.

        Args:
            target_node (ClusterNode, optional): Target node to replace the default
            node. Defaults to None.
        """
        if target_node:
            self.nodes_manager.default_node = target_node
        else:
            curr_node = self.get_default_node()
            primaries = [node for node in self.get_primaries() if node != curr_node]
            if primaries:
                self.nodes_manager.default_node = random.choice(primaries)
            else:
                replicas = [node for node in self.get_replicas() if node != curr_node]
                if replicas:
                    self.nodes_manager.default_node = random.choice(replicas)