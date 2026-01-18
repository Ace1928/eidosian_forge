import copy
import random
import socket
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union
from aiokeydb.v1.core import CaseInsensitiveDict, PubSub, KeyDB, parse_scan
from aiokeydb.v1.commands import READ_COMMANDS, CommandsParser, KeyDBClusterCommands
from aiokeydb.v1.connection import ConnectionPool, DefaultParser, Encoder, parse_url
from aiokeydb.v1.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.lock import Lock
from aiokeydb.v1.utils import (
class AbstractKeyDBCluster:
    KeyDBClusterRequestTTL = 16
    PRIMARIES = 'primaries'
    REPLICAS = 'replicas'
    ALL_NODES = 'all'
    RANDOM = 'random'
    DEFAULT_NODE = 'default-node'
    NODE_FLAGS = {PRIMARIES, REPLICAS, ALL_NODES, RANDOM, DEFAULT_NODE}
    COMMAND_FLAGS = dict_merge(list_keys_to_dict(['ACL CAT', 'ACL DELUSER', 'ACL DRYRUN', 'ACL GENPASS', 'ACL GETUSER', 'ACL HELP', 'ACL LIST', 'ACL LOG', 'ACL LOAD', 'ACL SAVE', 'ACL SETUSER', 'ACL USERS', 'ACL WHOAMI', 'AUTH', 'CLIENT LIST', 'CLIENT SETNAME', 'CLIENT GETNAME', 'CONFIG SET', 'CONFIG REWRITE', 'CONFIG RESETSTAT', 'TIME', 'PUBSUB CHANNELS', 'PUBSUB NUMPAT', 'PUBSUB NUMSUB', 'PING', 'INFO', 'SHUTDOWN', 'KEYS', 'DBSIZE', 'BGSAVE', 'SLOWLOG GET', 'SLOWLOG LEN', 'SLOWLOG RESET', 'WAIT', 'SAVE', 'MEMORY PURGE', 'MEMORY MALLOC-STATS', 'MEMORY STATS', 'LASTSAVE', 'CLIENT TRACKINGINFO', 'CLIENT PAUSE', 'CLIENT UNPAUSE', 'CLIENT UNBLOCK', 'CLIENT ID', 'CLIENT REPLY', 'CLIENT GETREDIR', 'CLIENT INFO', 'CLIENT KILL', 'READONLY', 'READWRITE', 'CLUSTER INFO', 'CLUSTER MEET', 'CLUSTER NODES', 'CLUSTER REPLICAS', 'CLUSTER RESET', 'CLUSTER SET-CONFIG-EPOCH', 'CLUSTER SLOTS', 'CLUSTER SHARDS', 'CLUSTER COUNT-FAILURE-REPORTS', 'CLUSTER KEYSLOT', 'COMMAND', 'COMMAND COUNT', 'COMMAND LIST', 'COMMAND GETKEYS', 'CONFIG GET', 'DEBUG', 'RANDOMKEY', 'READONLY', 'READWRITE', 'TIME', 'GRAPH.CONFIG'], DEFAULT_NODE), list_keys_to_dict(['FLUSHALL', 'FLUSHDB', 'FUNCTION DELETE', 'FUNCTION FLUSH', 'FUNCTION LIST', 'FUNCTION LOAD', 'FUNCTION RESTORE', 'SCAN', 'SCRIPT EXISTS', 'SCRIPT FLUSH', 'SCRIPT LOAD'], PRIMARIES), list_keys_to_dict(['FUNCTION DUMP'], RANDOM), list_keys_to_dict(['CLUSTER COUNTKEYSINSLOT', 'CLUSTER DELSLOTS', 'CLUSTER DELSLOTSRANGE', 'CLUSTER GETKEYSINSLOT', 'CLUSTER SETSLOT'], SLOT_ID))
    SEARCH_COMMANDS = (['FT.CREATE', 'FT.SEARCH', 'FT.AGGREGATE', 'FT.EXPLAIN', 'FT.EXPLAINCLI', 'FT,PROFILE', 'FT.ALTER', 'FT.DROPINDEX', 'FT.ALIASADD', 'FT.ALIASUPDATE', 'FT.ALIASDEL', 'FT.TAGVALS', 'FT.SUGADD', 'FT.SUGGET', 'FT.SUGDEL', 'FT.SUGLEN', 'FT.SYNUPDATE', 'FT.SYNDUMP', 'FT.SPELLCHECK', 'FT.DICTADD', 'FT.DICTDEL', 'FT.DICTDUMP', 'FT.INFO', 'FT._LIST', 'FT.CONFIG', 'FT.ADD', 'FT.DEL', 'FT.DROP', 'FT.GET', 'FT.MGET', 'FT.SYNADD'],)
    CLUSTER_COMMANDS_RESPONSE_CALLBACKS = {'CLUSTER SLOTS': parse_cluster_slots, 'CLUSTER SHARDS': parse_cluster_shards}
    RESULT_CALLBACKS = dict_merge(list_keys_to_dict(['PUBSUB NUMSUB'], parse_pubsub_numsub), list_keys_to_dict(['PUBSUB NUMPAT'], lambda command, res: sum(list(res.values()))), list_keys_to_dict(['KEYS', 'PUBSUB CHANNELS'], merge_result), list_keys_to_dict(['PING', 'CONFIG SET', 'CONFIG REWRITE', 'CONFIG RESETSTAT', 'CLIENT SETNAME', 'BGSAVE', 'SLOWLOG RESET', 'SAVE', 'MEMORY PURGE', 'CLIENT PAUSE', 'CLIENT UNPAUSE'], lambda command, res: all(res.values()) if isinstance(res, dict) else res), list_keys_to_dict(['DBSIZE', 'WAIT'], lambda command, res: sum(res.values()) if isinstance(res, dict) else res), list_keys_to_dict(['CLIENT UNBLOCK'], lambda command, res: 1 if sum(res.values()) > 0 else 0), list_keys_to_dict(['SCAN'], parse_scan_result), list_keys_to_dict(['SCRIPT LOAD'], lambda command, res: list(res.values()).pop()), list_keys_to_dict(['SCRIPT EXISTS'], lambda command, res: [all(k) for k in zip(*res.values())]), list_keys_to_dict(['SCRIPT FLUSH'], lambda command, res: all(res.values())))
    ERRORS_ALLOW_RETRY = (ConnectionError, TimeoutError, ClusterDownError)