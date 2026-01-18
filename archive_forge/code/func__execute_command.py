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
def _execute_command(self, target_node, *args, **kwargs):
    """
        Send a command to a node in the cluster
        """
    command = args[0]
    redis_node = None
    connection = None
    redirect_addr = None
    asking = False
    moved = False
    ttl = int(self.RedisClusterRequestTTL)
    while ttl > 0:
        ttl -= 1
        try:
            if asking:
                target_node = self.get_node(node_name=redirect_addr)
            elif moved:
                slot = self.determine_slot(*args)
                target_node = self.nodes_manager.get_node_from_slot(slot, self.read_from_replicas and command in READ_COMMANDS)
                moved = False
            redis_node = self.get_redis_connection(target_node)
            connection = get_connection(redis_node, *args, **kwargs)
            if asking:
                connection.send_command('ASKING')
                redis_node.parse_response(connection, 'ASKING', **kwargs)
                asking = False
            connection.send_command(*args)
            response = redis_node.parse_response(connection, command, **kwargs)
            if command in self.cluster_response_callbacks:
                response = self.cluster_response_callbacks[command](response, **kwargs)
            return response
        except AuthenticationError:
            raise
        except (ConnectionError, TimeoutError) as e:
            if connection is not None:
                connection.disconnect()
            self.nodes_manager.startup_nodes.pop(target_node.name, None)
            target_node.redis_connection = None
            self.nodes_manager.initialize()
            raise e
        except MovedError as e:
            self.reinitialize_counter += 1
            if self._should_reinitialized():
                self.nodes_manager.initialize()
                self.reinitialize_counter = 0
            else:
                self.nodes_manager.update_moved_exception(e)
            moved = True
        except TryAgainError:
            if ttl < self.RedisClusterRequestTTL / 2:
                time.sleep(0.05)
        except AskError as e:
            redirect_addr = get_node_name(host=e.host, port=e.port)
            asking = True
        except ClusterDownError as e:
            time.sleep(0.25)
            self.nodes_manager.initialize()
            raise e
        except ResponseError:
            raise
        except Exception as e:
            if connection:
                connection.disconnect()
            raise e
        finally:
            if connection is not None:
                redis_node.connection_pool.release(connection)
    raise ClusterError('TTL exhausted.')