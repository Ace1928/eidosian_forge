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
class RedisCluster(AbstractRedis, AbstractRedisCluster, AsyncRedisClusterCommands):
    """
    Create a new RedisCluster client.

    Pass one of parameters:

      - `host` & `port`
      - `startup_nodes`

    | Use ``await`` :meth:`initialize` to find cluster nodes & create connections.
    | Use ``await`` :meth:`close` to disconnect connections & close client.

    Many commands support the target_nodes kwarg. It can be one of the
    :attr:`NODE_FLAGS`:

      - :attr:`PRIMARIES`
      - :attr:`REPLICAS`
      - :attr:`ALL_NODES`
      - :attr:`RANDOM`
      - :attr:`DEFAULT_NODE`

    Note: This client is not thread/process/fork safe.

    :param host:
        | Can be used to point to a startup node
    :param port:
        | Port used if **host** is provided
    :param startup_nodes:
        | :class:`~.ClusterNode` to used as a startup node
    :param require_full_coverage:
        | When set to ``False``: the client will not require a full coverage of
          the slots. However, if not all slots are covered, and at least one node
          has ``cluster-require-full-coverage`` set to ``yes``, the server will throw
          a :class:`~.ClusterDownError` for some key-based commands.
        | When set to ``True``: all slots must be covered to construct the cluster
          client. If not all slots are covered, :class:`~.RedisClusterException` will be
          thrown.
        | See:
          https://redis.io/docs/manual/scaling/#redis-cluster-configuration-parameters
    :param read_from_replicas:
        | Enable read from replicas in READONLY mode. You can read possibly stale data.
          When set to true, read commands will be assigned between the primary and
          its replications in a Round-Robin manner.
    :param reinitialize_steps:
        | Specifies the number of MOVED errors that need to occur before reinitializing
          the whole cluster topology. If a MOVED error occurs and the cluster does not
          need to be reinitialized on this current error handling, only the MOVED slot
          will be patched with the redirected node.
          To reinitialize the cluster on every MOVED error, set reinitialize_steps to 1.
          To avoid reinitializing the cluster on moved errors, set reinitialize_steps to
          0.
    :param cluster_error_retry_attempts:
        | Number of times to retry before raising an error when :class:`~.TimeoutError`
          or :class:`~.ConnectionError` or :class:`~.ClusterDownError` are encountered
    :param connection_error_retry_attempts:
        | Number of times to retry before reinitializing when :class:`~.TimeoutError`
          or :class:`~.ConnectionError` are encountered.
          The default backoff strategy will be set if Retry object is not passed (see
          default_backoff in backoff.py). To change it, pass a custom Retry object
          using the "retry" keyword.
    :param max_connections:
        | Maximum number of connections per node. If there are no free connections & the
          maximum number of connections are already created, a
          :class:`~.MaxConnectionsError` is raised. This error may be retried as defined
          by :attr:`connection_error_retry_attempts`
    :param address_remap:
        | An optional callable which, when provided with an internal network
          address of a node, e.g. a `(host, port)` tuple, will return the address
          where the node is reachable.  This can be used to map the addresses at
          which the nodes _think_ they are, to addresses at which a client may
          reach them, such as when they sit behind a proxy.

    | Rest of the arguments will be passed to the
      :class:`~redis.asyncio.connection.Connection` instances when created

    :raises RedisClusterException:
        if any arguments are invalid or unknown. Eg:

        - `db` != 0 or None
        - `path` argument for unix socket connection
        - none of the `host`/`port` & `startup_nodes` were provided

    """

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> 'RedisCluster':
        """
        Return a Redis client object configured from the given URL.

        For example::

            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0

        Three URL schemes are supported:

        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>

        The username, password, hostname, path and all querystring values are passed
        through ``urllib.parse.unquote`` in order to replace any percent-encoded values
        with their corresponding characters.

        All querystring options are cast to their appropriate Python types. Boolean
        arguments can be specified with string values "True"/"False" or "Yes"/"No".
        Values that cannot be properly cast cause a ``ValueError`` to be raised. Once
        parsed, the querystring arguments and keyword arguments are passed to
        :class:`~redis.asyncio.connection.Connection` when created.
        In the case of conflicting arguments, querystring arguments are used.
        """
        kwargs.update(parse_url(url))
        if kwargs.pop('connection_class', None) is SSLConnection:
            kwargs['ssl'] = True
        return cls(**kwargs)
    __slots__ = ('_initialize', '_lock', 'cluster_error_retry_attempts', 'command_flags', 'commands_parser', 'connection_error_retry_attempts', 'connection_kwargs', 'encoder', 'node_flags', 'nodes_manager', 'read_from_replicas', 'reinitialize_counter', 'reinitialize_steps', 'response_callbacks', 'result_callbacks')

    def __init__(self, host: Optional[str]=None, port: Union[str, int]=6379, startup_nodes: Optional[List['ClusterNode']]=None, require_full_coverage: bool=True, read_from_replicas: bool=False, reinitialize_steps: int=5, cluster_error_retry_attempts: int=3, connection_error_retry_attempts: int=3, max_connections: int=2 ** 31, db: Union[str, int]=0, path: Optional[str]=None, credential_provider: Optional[CredentialProvider]=None, username: Optional[str]=None, password: Optional[str]=None, client_name: Optional[str]=None, lib_name: Optional[str]='redis-py', lib_version: Optional[str]=get_lib_version(), encoding: str='utf-8', encoding_errors: str='strict', decode_responses: bool=False, health_check_interval: float=0, socket_connect_timeout: Optional[float]=None, socket_keepalive: bool=False, socket_keepalive_options: Optional[Mapping[int, Union[int, bytes]]]=None, socket_timeout: Optional[float]=None, retry: Optional['Retry']=None, retry_on_error: Optional[List[Type[Exception]]]=None, ssl: bool=False, ssl_ca_certs: Optional[str]=None, ssl_ca_data: Optional[str]=None, ssl_cert_reqs: str='required', ssl_certfile: Optional[str]=None, ssl_check_hostname: bool=False, ssl_keyfile: Optional[str]=None, ssl_min_version: Optional[ssl.TLSVersion]=None, protocol: Optional[int]=2, address_remap: Optional[Callable[[str, int], Tuple[str, int]]]=None) -> None:
        if db:
            raise RedisClusterException("Argument 'db' must be 0 or None in cluster mode")
        if path:
            raise RedisClusterException('Unix domain socket is not supported in cluster mode')
        if (not host or not port) and (not startup_nodes):
            raise RedisClusterException('RedisCluster requires at least one node to discover the cluster.\nPlease provide one of the following or use RedisCluster.from_url:\n   - host and port: RedisCluster(host="localhost", port=6379)\n   - startup_nodes: RedisCluster(startup_nodes=[ClusterNode("localhost", 6379), ClusterNode("localhost", 6380)])')
        kwargs: Dict[str, Any] = {'max_connections': max_connections, 'connection_class': Connection, 'parser_class': ClusterParser, 'credential_provider': credential_provider, 'username': username, 'password': password, 'client_name': client_name, 'lib_name': lib_name, 'lib_version': lib_version, 'encoding': encoding, 'encoding_errors': encoding_errors, 'decode_responses': decode_responses, 'health_check_interval': health_check_interval, 'socket_connect_timeout': socket_connect_timeout, 'socket_keepalive': socket_keepalive, 'socket_keepalive_options': socket_keepalive_options, 'socket_timeout': socket_timeout, 'retry': retry, 'protocol': protocol}
        if ssl:
            kwargs.update({'connection_class': SSLConnection, 'ssl_ca_certs': ssl_ca_certs, 'ssl_ca_data': ssl_ca_data, 'ssl_cert_reqs': ssl_cert_reqs, 'ssl_certfile': ssl_certfile, 'ssl_check_hostname': ssl_check_hostname, 'ssl_keyfile': ssl_keyfile, 'ssl_min_version': ssl_min_version})
        if read_from_replicas:
            kwargs['redis_connect_func'] = self.on_connect
        self.retry = retry
        if retry or retry_on_error or connection_error_retry_attempts > 0:
            self.retry = retry or Retry(default_backoff(), connection_error_retry_attempts)
            if not retry_on_error:
                retry_on_error = [ConnectionError, TimeoutError]
            self.retry.update_supported_errors(retry_on_error)
            kwargs.update({'retry': self.retry})
        kwargs['response_callbacks'] = _RedisCallbacks.copy()
        if kwargs.get('protocol') in ['3', 3]:
            kwargs['response_callbacks'].update(_RedisCallbacksRESP3)
        else:
            kwargs['response_callbacks'].update(_RedisCallbacksRESP2)
        self.connection_kwargs = kwargs
        if startup_nodes:
            passed_nodes = []
            for node in startup_nodes:
                passed_nodes.append(ClusterNode(node.host, node.port, **self.connection_kwargs))
            startup_nodes = passed_nodes
        else:
            startup_nodes = []
        if host and port:
            startup_nodes.append(ClusterNode(host, port, **self.connection_kwargs))
        self.nodes_manager = NodesManager(startup_nodes, require_full_coverage, kwargs, address_remap=address_remap)
        self.encoder = Encoder(encoding, encoding_errors, decode_responses)
        self.read_from_replicas = read_from_replicas
        self.reinitialize_steps = reinitialize_steps
        self.cluster_error_retry_attempts = cluster_error_retry_attempts
        self.connection_error_retry_attempts = connection_error_retry_attempts
        self.reinitialize_counter = 0
        self.commands_parser = AsyncCommandsParser()
        self.node_flags = self.__class__.NODE_FLAGS.copy()
        self.command_flags = self.__class__.COMMAND_FLAGS.copy()
        self.response_callbacks = kwargs['response_callbacks']
        self.result_callbacks = self.__class__.RESULT_CALLBACKS.copy()
        self.result_callbacks['CLUSTER SLOTS'] = lambda cmd, res, **kwargs: parse_cluster_slots(list(res.values())[0], **kwargs)
        self._initialize = True
        self._lock: Optional[asyncio.Lock] = None

    async def initialize(self) -> 'RedisCluster':
        """Get all nodes from startup nodes & creates connections if not initialized."""
        if self._initialize:
            if not self._lock:
                self._lock = asyncio.Lock()
            async with self._lock:
                if self._initialize:
                    try:
                        await self.nodes_manager.initialize()
                        await self.commands_parser.initialize(self.nodes_manager.default_node)
                        self._initialize = False
                    except BaseException:
                        await self.nodes_manager.aclose()
                        await self.nodes_manager.aclose('startup_nodes')
                        raise
        return self

    async def aclose(self) -> None:
        """Close all connections & client if initialized."""
        if not self._initialize:
            if not self._lock:
                self._lock = asyncio.Lock()
            async with self._lock:
                if not self._initialize:
                    self._initialize = True
                    await self.nodes_manager.aclose()
                    await self.nodes_manager.aclose('startup_nodes')

    @deprecated_function(version='5.0.0', reason='Use aclose() instead', name='close')
    async def close(self) -> None:
        """alias for aclose() for backwards compatibility"""
        await self.aclose()

    async def __aenter__(self) -> 'RedisCluster':
        return await self.initialize()

    async def __aexit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
        await self.aclose()

    def __await__(self) -> Generator[Any, None, 'RedisCluster']:
        return self.initialize().__await__()
    _DEL_MESSAGE = 'Unclosed RedisCluster client'

    def __del__(self, _warn: Any=warnings.warn, _grl: Any=asyncio.get_running_loop) -> None:
        if hasattr(self, '_initialize') and (not self._initialize):
            _warn(f'{self._DEL_MESSAGE} {self!r}', ResourceWarning, source=self)
            try:
                context = {'client': self, 'message': self._DEL_MESSAGE}
                _grl().call_exception_handler(context)
            except RuntimeError:
                pass

    async def on_connect(self, connection: Connection) -> None:
        await connection.on_connect()
        await connection.send_command('READONLY')
        if str_if_bytes(await connection.read_response()) != 'OK':
            raise ConnectionError('READONLY command failed')

    def get_nodes(self) -> List['ClusterNode']:
        """Get all nodes of the cluster."""
        return list(self.nodes_manager.nodes_cache.values())

    def get_primaries(self) -> List['ClusterNode']:
        """Get the primary nodes of the cluster."""
        return self.nodes_manager.get_nodes_by_server_type(PRIMARY)

    def get_replicas(self) -> List['ClusterNode']:
        """Get the replica nodes of the cluster."""
        return self.nodes_manager.get_nodes_by_server_type(REPLICA)

    def get_random_node(self) -> 'ClusterNode':
        """Get a random node of the cluster."""
        return random.choice(list(self.nodes_manager.nodes_cache.values()))

    def get_default_node(self) -> 'ClusterNode':
        """Get the default node of the client."""
        return self.nodes_manager.default_node

    def set_default_node(self, node: 'ClusterNode') -> None:
        """
        Set the default node of the client.

        :raises DataError: if None is passed or node does not exist in cluster.
        """
        if not node or not self.get_node(node_name=node.name):
            raise DataError('The requested node does not exist in the cluster.')
        self.nodes_manager.default_node = node

    def get_node(self, host: Optional[str]=None, port: Optional[int]=None, node_name: Optional[str]=None) -> Optional['ClusterNode']:
        """Get node by (host, port) or node_name."""
        return self.nodes_manager.get_node(host, port, node_name)

    def get_node_from_key(self, key: str, replica: bool=False) -> Optional['ClusterNode']:
        """
        Get the cluster node corresponding to the provided key.

        :param key:
        :param replica:
            | Indicates if a replica should be returned
            |
              None will returned if no replica holds this key

        :raises SlotNotCoveredError: if the key is not covered by any slot.
        """
        slot = self.keyslot(key)
        slot_cache = self.nodes_manager.slots_cache.get(slot)
        if not slot_cache:
            raise SlotNotCoveredError(f'Slot "{slot}" is not covered by the cluster.')
        if replica:
            if len(self.nodes_manager.slots_cache[slot]) < 2:
                return None
            node_idx = 1
        else:
            node_idx = 0
        return slot_cache[node_idx]

    def keyslot(self, key: EncodableT) -> int:
        """
        Find the keyslot for a given key.

        See: https://redis.io/docs/manual/scaling/#redis-cluster-data-sharding
        """
        return key_slot(self.encoder.encode(key))

    def get_encoder(self) -> Encoder:
        """Get the encoder object of the client."""
        return self.encoder

    def get_connection_kwargs(self) -> Dict[str, Optional[Any]]:
        """Get the kwargs passed to :class:`~redis.asyncio.connection.Connection`."""
        return self.connection_kwargs

    def get_retry(self) -> Optional['Retry']:
        return self.retry

    def set_retry(self, retry: 'Retry') -> None:
        self.retry = retry
        for node in self.get_nodes():
            node.connection_kwargs.update({'retry': retry})
            for conn in node._connections:
                conn.retry = retry

    def set_response_callback(self, command: str, callback: ResponseCallbackT) -> None:
        """Set a custom response callback."""
        self.response_callbacks[command] = callback

    async def _determine_nodes(self, command: str, *args: Any, node_flag: Optional[str]=None) -> List['ClusterNode']:
        if not node_flag:
            node_flag = self.command_flags.get(command)
        if node_flag in self.node_flags:
            if node_flag == self.__class__.DEFAULT_NODE:
                return [self.nodes_manager.default_node]
            if node_flag == self.__class__.PRIMARIES:
                return self.nodes_manager.get_nodes_by_server_type(PRIMARY)
            if node_flag == self.__class__.REPLICAS:
                return self.nodes_manager.get_nodes_by_server_type(REPLICA)
            if node_flag == self.__class__.ALL_NODES:
                return list(self.nodes_manager.nodes_cache.values())
            if node_flag == self.__class__.RANDOM:
                return [random.choice(list(self.nodes_manager.nodes_cache.values()))]
        return [self.nodes_manager.get_node_from_slot(await self._determine_slot(command, *args), self.read_from_replicas and command in READ_COMMANDS)]

    async def _determine_slot(self, command: str, *args: Any) -> int:
        if self.command_flags.get(command) == SLOT_ID:
            return int(args[0])
        if command.upper() in ('EVAL', 'EVALSHA'):
            if len(args) < 2:
                raise RedisClusterException(f'Invalid args in command: {(command, *args)}')
            keys = args[2:2 + int(args[1])]
            if not keys:
                return random.randrange(0, REDIS_CLUSTER_HASH_SLOTS)
        else:
            keys = await self.commands_parser.get_keys(command, *args)
            if not keys:
                if command.upper() in ('FCALL', 'FCALL_RO'):
                    return random.randrange(0, REDIS_CLUSTER_HASH_SLOTS)
                raise RedisClusterException(f'No way to dispatch this command to Redis Cluster. Missing key.\nYou can execute the command by specifying target nodes.\nCommand: {args}')
        if len(keys) == 1:
            return self.keyslot(keys[0])
        slots = {self.keyslot(key) for key in keys}
        if len(slots) != 1:
            raise RedisClusterException(f'{command} - all keys must map to the same key slot')
        return slots.pop()

    def _is_node_flag(self, target_nodes: Any) -> bool:
        return isinstance(target_nodes, str) and target_nodes in self.node_flags

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

    async def execute_command(self, *args: EncodableT, **kwargs: Any) -> Any:
        """
        Execute a raw command on the appropriate cluster node or target_nodes.

        It will retry the command as specified by :attr:`cluster_error_retry_attempts` &
        then raise an exception.

        :param args:
            | Raw command args
        :param kwargs:

            - target_nodes: :attr:`NODE_FLAGS` or :class:`~.ClusterNode`
              or List[:class:`~.ClusterNode`] or Dict[Any, :class:`~.ClusterNode`]
            - Rest of the kwargs are passed to the Redis connection

        :raises RedisClusterException: if target_nodes is not provided & the command
            can't be mapped to a slot
        """
        command = args[0]
        target_nodes = []
        target_nodes_specified = False
        retry_attempts = self.cluster_error_retry_attempts
        passed_targets = kwargs.pop('target_nodes', None)
        if passed_targets and (not self._is_node_flag(passed_targets)):
            target_nodes = self._parse_target_nodes(passed_targets)
            target_nodes_specified = True
            retry_attempts = 0
        execute_attempts = 1 + retry_attempts
        for _ in range(execute_attempts):
            if self._initialize:
                await self.initialize()
                if len(target_nodes) == 1 and target_nodes[0] == self.get_default_node():
                    self.replace_default_node()
            try:
                if not target_nodes_specified:
                    target_nodes = await self._determine_nodes(*args, node_flag=passed_targets)
                    if not target_nodes:
                        raise RedisClusterException(f'No targets were found to execute {args} command on')
                if len(target_nodes) == 1:
                    ret = await self._execute_command(target_nodes[0], *args, **kwargs)
                    if command in self.result_callbacks:
                        return self.result_callbacks[command](command, {target_nodes[0].name: ret}, **kwargs)
                    return ret
                else:
                    keys = [node.name for node in target_nodes]
                    values = await asyncio.gather(*(asyncio.create_task(self._execute_command(node, *args, **kwargs)) for node in target_nodes))
                    if command in self.result_callbacks:
                        return self.result_callbacks[command](command, dict(zip(keys, values)), **kwargs)
                    return dict(zip(keys, values))
            except Exception as e:
                if retry_attempts > 0 and type(e) in self.__class__.ERRORS_ALLOW_RETRY:
                    retry_attempts -= 1
                    continue
                else:
                    raise e

    async def _execute_command(self, target_node: 'ClusterNode', *args: Union[KeyT, EncodableT], **kwargs: Any) -> Any:
        asking = moved = False
        redirect_addr = None
        ttl = self.RedisClusterRequestTTL
        while ttl > 0:
            ttl -= 1
            try:
                if asking:
                    target_node = self.get_node(node_name=redirect_addr)
                    await target_node.execute_command('ASKING')
                    asking = False
                elif moved:
                    slot = await self._determine_slot(*args)
                    target_node = self.nodes_manager.get_node_from_slot(slot, self.read_from_replicas and args[0] in READ_COMMANDS)
                    moved = False
                return await target_node.execute_command(*args, **kwargs)
            except (BusyLoadingError, MaxConnectionsError):
                raise
            except (ConnectionError, TimeoutError):
                self.nodes_manager.startup_nodes.pop(target_node.name, None)
                await self.aclose()
                raise
            except ClusterDownError:
                await self.aclose()
                await asyncio.sleep(0.25)
                raise
            except MovedError as e:
                self.reinitialize_counter += 1
                if self.reinitialize_steps and self.reinitialize_counter % self.reinitialize_steps == 0:
                    await self.aclose()
                    self.reinitialize_counter = 0
                else:
                    self.nodes_manager._moved_exception = e
                moved = True
            except AskError as e:
                redirect_addr = get_node_name(host=e.host, port=e.port)
                asking = True
            except TryAgainError:
                if ttl < self.RedisClusterRequestTTL / 2:
                    await asyncio.sleep(0.05)
        raise ClusterError('TTL exhausted.')

    def pipeline(self, transaction: Optional[Any]=None, shard_hint: Optional[Any]=None) -> 'ClusterPipeline':
        """
        Create & return a new :class:`~.ClusterPipeline` object.

        Cluster implementation of pipeline does not support transaction or shard_hint.

        :raises RedisClusterException: if transaction or shard_hint are truthy values
        """
        if shard_hint:
            raise RedisClusterException('shard_hint is deprecated in cluster mode')
        if transaction:
            raise RedisClusterException('transaction is deprecated in cluster mode')
        return ClusterPipeline(self)

    def lock(self, name: KeyT, timeout: Optional[float]=None, sleep: float=0.1, blocking: bool=True, blocking_timeout: Optional[float]=None, lock_class: Optional[Type[Lock]]=None, thread_local: bool=True) -> Lock:
        """
        Return a new Lock object using key ``name`` that mimics
        the behavior of threading.Lock.

        If specified, ``timeout`` indicates a maximum life for the lock.
        By default, it will remain locked until release() is called.

        ``sleep`` indicates the amount of time to sleep per loop iteration
        when the lock is in blocking mode and another client is currently
        holding the lock.

        ``blocking`` indicates whether calling ``acquire`` should block until
        the lock has been acquired or to fail immediately, causing ``acquire``
        to return False and the lock not being acquired. Defaults to True.
        Note this value can be overridden by passing a ``blocking``
        argument to ``acquire``.

        ``blocking_timeout`` indicates the maximum amount of time in seconds to
        spend trying to acquire the lock. A value of ``None`` indicates
        continue trying forever. ``blocking_timeout`` can be specified as a
        float or integer, both representing the number of seconds to wait.

        ``lock_class`` forces the specified lock implementation. Note that as
        of redis-py 3.0, the only lock class we implement is ``Lock`` (which is
        a Lua-based lock). So, it's unlikely you'll need this parameter, unless
        you have created your own custom lock class.

        ``thread_local`` indicates whether the lock token is placed in
        thread-local storage. By default, the token is placed in thread local
        storage so that a thread only sees its token, not a token set by
        another thread. Consider the following timeline:

            time: 0, thread-1 acquires `my-lock`, with a timeout of 5 seconds.
                     thread-1 sets the token to "abc"
            time: 1, thread-2 blocks trying to acquire `my-lock` using the
                     Lock instance.
            time: 5, thread-1 has not yet completed. redis expires the lock
                     key.
            time: 5, thread-2 acquired `my-lock` now that it's available.
                     thread-2 sets the token to "xyz"
            time: 6, thread-1 finishes its work and calls release(). if the
                     token is *not* stored in thread local storage, then
                     thread-1 would see the token value as "xyz" and would be
                     able to successfully release the thread-2's lock.

        In some use cases it's necessary to disable thread local storage. For
        example, if you have code where one thread acquires a lock and passes
        that lock instance to a worker thread to release later. If thread
        local storage isn't disabled in this case, the worker thread won't see
        the token set by the thread that acquired the lock. Our assumption
        is that these cases aren't common and as such default to using
        thread local storage."""
        if lock_class is None:
            lock_class = Lock
        return lock_class(self, name, timeout=timeout, sleep=sleep, blocking=blocking, blocking_timeout=blocking_timeout, thread_local=thread_local)