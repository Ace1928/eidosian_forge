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
def _get_or_create_cluster_node(self, host, port, role, tmp_nodes_cache):
    node_name = get_node_name(host, port)
    target_node = tmp_nodes_cache.get(node_name)
    if target_node is None:
        target_node = self.nodes_cache.get(node_name)
        if target_node is None or target_node.redis_connection is None:
            target_node = ClusterNode(host, port, role)
        if target_node.server_type != role:
            target_node.server_type = role
    return target_node