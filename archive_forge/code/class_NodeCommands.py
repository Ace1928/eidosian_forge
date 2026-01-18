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
class NodeCommands:
    """ """

    def __init__(self, parse_response, connection_pool, connection):
        """ """
        self.parse_response = parse_response
        self.connection_pool = connection_pool
        self.connection = connection
        self.commands = []

    def append(self, c):
        """ """
        self.commands.append(c)

    def write(self):
        """
        Code borrowed from Redis so it can be fixed
        """
        connection = self.connection
        commands = self.commands
        for c in commands:
            c.result = None
        try:
            connection.send_packed_command(connection.pack_commands([c.args for c in commands]))
        except (ConnectionError, TimeoutError) as e:
            for c in commands:
                c.result = e

    def read(self):
        """ """
        connection = self.connection
        for c in self.commands:
            if c.result is None:
                try:
                    c.result = self.parse_response(connection, c.args[0], **c.options)
                except (ConnectionError, TimeoutError) as e:
                    for c in self.commands:
                        c.result = e
                    return
                except RedisError:
                    c.result = sys.exc_info()[1]