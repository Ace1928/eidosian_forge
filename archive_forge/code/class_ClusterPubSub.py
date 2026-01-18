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
class ClusterPubSub(PubSub):
    """
    Wrapper for PubSub class.

    IMPORTANT: before using ClusterPubSub, read about the known limitations
    with pubsub in Cluster mode and learn how to workaround them:
    https://redis-py-cluster.readthedocs.io/en/stable/pubsub.html
    """

    def __init__(self, redis_cluster, node=None, host=None, port=None, push_handler_func=None, **kwargs):
        """
        When a pubsub instance is created without specifying a node, a single
        node will be transparently chosen for the pubsub connection on the
        first command execution. The node will be determined by:
         1. Hashing the channel name in the request to find its keyslot
         2. Selecting a node that handles the keyslot: If read_from_replicas is
            set to true, a replica can be selected.

        :type redis_cluster: RedisCluster
        :type node: ClusterNode
        :type host: str
        :type port: int
        """
        self.node = None
        self.set_pubsub_node(redis_cluster, node, host, port)
        connection_pool = None if self.node is None else redis_cluster.get_redis_connection(self.node).connection_pool
        self.cluster = redis_cluster
        self.node_pubsub_mapping = {}
        self._pubsubs_generator = self._pubsubs_generator()
        super().__init__(connection_pool=connection_pool, encoder=redis_cluster.encoder, push_handler_func=push_handler_func, **kwargs)

    def set_pubsub_node(self, cluster, node=None, host=None, port=None):
        """
        The pubsub node will be set according to the passed node, host and port
        When none of the node, host, or port are specified - the node is set
        to None and will be determined by the keyslot of the channel in the
        first command to be executed.
        RedisClusterException will be thrown if the passed node does not exist
        in the cluster.
        If host is passed without port, or vice versa, a DataError will be
        thrown.
        :type cluster: RedisCluster
        :type node: ClusterNode
        :type host: str
        :type port: int
        """
        if node is not None:
            self._raise_on_invalid_node(cluster, node, node.host, node.port)
            pubsub_node = node
        elif host is not None and port is not None:
            node = cluster.get_node(host=host, port=port)
            self._raise_on_invalid_node(cluster, node, host, port)
            pubsub_node = node
        elif any([host, port]) is True:
            raise DataError('Passing a host requires passing a port, and vice versa')
        else:
            pubsub_node = None
        self.node = pubsub_node

    def get_pubsub_node(self):
        """
        Get the node that is being used as the pubsub connection
        """
        return self.node

    def _raise_on_invalid_node(self, redis_cluster, node, host, port):
        """
        Raise a RedisClusterException if the node is None or doesn't exist in
        the cluster.
        """
        if node is None or redis_cluster.get_node(node_name=node.name) is None:
            raise RedisClusterException(f"Node {host}:{port} doesn't exist in the cluster")

    def execute_command(self, *args):
        """
        Execute a subscribe/unsubscribe command.

        Taken code from redis-py and tweak to make it work within a cluster.
        """
        if self.connection is None:
            if self.connection_pool is None:
                if len(args) > 1:
                    channel = args[1]
                    slot = self.cluster.keyslot(channel)
                    node = self.cluster.nodes_manager.get_node_from_slot(slot, self.cluster.read_from_replicas)
                else:
                    node = self.cluster.get_random_node()
                self.node = node
                redis_connection = self.cluster.get_redis_connection(node)
                self.connection_pool = redis_connection.connection_pool
            self.connection = self.connection_pool.get_connection('pubsub', self.shard_hint)
            self.connection.register_connect_callback(self.on_connect)
            if self.push_handler_func is not None and (not HIREDIS_AVAILABLE):
                self.connection._parser.set_push_handler(self.push_handler_func)
        connection = self.connection
        self._execute(connection, connection.send_command, *args)

    def _get_node_pubsub(self, node):
        try:
            return self.node_pubsub_mapping[node.name]
        except KeyError:
            pubsub = node.redis_connection.pubsub(push_handler_func=self.push_handler_func)
            self.node_pubsub_mapping[node.name] = pubsub
            return pubsub

    def _sharded_message_generator(self):
        for _ in range(len(self.node_pubsub_mapping)):
            pubsub = next(self._pubsubs_generator)
            message = pubsub.get_message()
            if message is not None:
                return message
        return None

    def _pubsubs_generator(self):
        while True:
            for pubsub in self.node_pubsub_mapping.values():
                yield pubsub

    def get_sharded_message(self, ignore_subscribe_messages=False, timeout=0.0, target_node=None):
        if target_node:
            message = self.node_pubsub_mapping[target_node.name].get_message(ignore_subscribe_messages=ignore_subscribe_messages, timeout=timeout)
        else:
            message = self._sharded_message_generator()
        if message is None:
            return None
        elif str_if_bytes(message['type']) == 'sunsubscribe':
            if message['channel'] in self.pending_unsubscribe_shard_channels:
                self.pending_unsubscribe_shard_channels.remove(message['channel'])
                self.shard_channels.pop(message['channel'], None)
                node = self.cluster.get_node_from_key(message['channel'])
                if self.node_pubsub_mapping[node.name].subscribed is False:
                    self.node_pubsub_mapping.pop(node.name)
        if not self.channels and (not self.patterns) and (not self.shard_channels):
            self.subscribed_event.clear()
        if self.ignore_subscribe_messages or ignore_subscribe_messages:
            return None
        return message

    def ssubscribe(self, *args, **kwargs):
        if args:
            args = list_or_args(args[0], args[1:])
        s_channels = dict.fromkeys(args)
        s_channels.update(kwargs)
        for s_channel, handler in s_channels.items():
            node = self.cluster.get_node_from_key(s_channel)
            pubsub = self._get_node_pubsub(node)
            if handler:
                pubsub.ssubscribe(**{s_channel: handler})
            else:
                pubsub.ssubscribe(s_channel)
            self.shard_channels.update(pubsub.shard_channels)
            self.pending_unsubscribe_shard_channels.difference_update(self._normalize_keys({s_channel: None}))
            if pubsub.subscribed and (not self.subscribed):
                self.subscribed_event.set()
                self.health_check_response_counter = 0

    def sunsubscribe(self, *args):
        if args:
            args = list_or_args(args[0], args[1:])
        else:
            args = self.shard_channels
        for s_channel in args:
            node = self.cluster.get_node_from_key(s_channel)
            p = self._get_node_pubsub(node)
            p.sunsubscribe(s_channel)
            self.pending_unsubscribe_shard_channels.update(p.pending_unsubscribe_shard_channels)

    def get_redis_connection(self):
        """
        Get the Redis connection of the pubsub connected node.
        """
        if self.node is not None:
            return self.node.redis_connection

    def disconnect(self):
        """
        Disconnect the pubsub connection.
        """
        if self.connection:
            self.connection.disconnect()
        for pubsub in self.node_pubsub_mapping.values():
            pubsub.connection.disconnect()