import asyncio
import collections
import random
import socket
import ssl
import warnings
from typing import (
from redis._parsers import AsyncCommandsParser, Encoder
from redis._parsers.helpers import (
from redis.asyncio.client import ResponseCallbackT
from redis.asyncio.connection import Connection, DefaultParser, SSLConnection, parse_url
from redis.asyncio.lock import Lock
from redis.asyncio.retry import Retry
from redis.backoff import default_backoff
from redis.client import EMPTY_RESPONSE, NEVER_DECODE, AbstractRedis
from redis.cluster import (
from redis.commands import READ_COMMANDS, AsyncRedisClusterCommands
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.credentials import CredentialProvider
from redis.exceptions import (
from redis.typing import AnyKeyT, EncodableT, KeyT
from redis.utils import (
def _parse_target_nodes(self, target_nodes: Any) -> List['ClusterNode']:
    if isinstance(target_nodes, list):
        nodes = target_nodes
    elif isinstance(target_nodes, ClusterNode):
        nodes = [target_nodes]
    elif isinstance(target_nodes, dict):
        nodes = list(target_nodes.values())
    else:
        raise TypeError(f'target_nodes type can be one of the following: node_flag (PRIMARIES, REPLICAS, RANDOM, ALL_NODES),ClusterNode, list<ClusterNode>, or dict<any, ClusterNode>. The passed type is {type(target_nodes)}')
    return nodes