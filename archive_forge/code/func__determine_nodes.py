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
def _determine_nodes(self, *args, **kwargs) -> List['ClusterNode']:
    command = args[0].upper()
    if len(args) >= 2 and f'{args[0]} {args[1]}'.upper() in self.command_flags:
        command = f'{args[0]} {args[1]}'.upper()
    nodes_flag = kwargs.pop('nodes_flag', None)
    if nodes_flag is not None:
        command_flag = nodes_flag
    else:
        command_flag = self.command_flags.get(command)
    if command_flag == self.__class__.RANDOM:
        return [self.get_random_node()]
    elif command_flag == self.__class__.PRIMARIES:
        return self.get_primaries()
    elif command_flag == self.__class__.REPLICAS:
        return self.get_replicas()
    elif command_flag == self.__class__.ALL_NODES:
        return self.get_nodes()
    elif command_flag == self.__class__.DEFAULT_NODE:
        return [self.nodes_manager.default_node]
    elif command in self.__class__.SEARCH_COMMANDS[0]:
        return [self.nodes_manager.default_node]
    else:
        slot = self.determine_slot(*args)
        node = self.nodes_manager.get_node_from_slot(slot, self.read_from_replicas and command in READ_COMMANDS)
        return [node]