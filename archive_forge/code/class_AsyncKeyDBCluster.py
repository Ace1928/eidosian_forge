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
class AsyncKeyDBCluster(AbstractKeyDB, AbstractKeyDBCluster, AsyncKeyDBClusterCommands):
    """
    Create a new AsyncKeyDBCluster client.

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
        | :class:`~.AsyncClusterNode` to used as a startup node
    :param require_full_coverage:
        | When set to ``False``: the client will not require a full coverage of the
          slots. However, if not all slots are covered, and at least one node has
          ``cluster-require-full-coverage`` set to ``yes``, the server will throw a
          :class:`~.ClusterDownError` for some key-based commands.
        | When set to ``True``: all slots must be covered to construct the cluster
          client. If not all slots are covered, :class:`~.KeyDBClusterException` will be
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
          or :class:`~.ConnectionError` are encountered
    :param max_connections:
        | Maximum number of connections per node. If there are no free connections & the
          maximum number of connections are already created, a
          :class:`~.MaxConnectionsError` is raised. This error may be retried as defined
          by :attr:`connection_error_retry_attempts`

    | Rest of the arguments will be passed to the
      :class:`~redis.asyncio.connection.AsyncConnection` instances when created

    :raises KeyDBClusterException:
        if any arguments are invalid or unknown. Eg:

        - `db` != 0 or None
        - `path` argument for unix socket connection
        - none of the `host`/`port` & `startup_nodes` were provided

    """

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> 'AsyncKeyDBCluster':
        """
        Return a KeyDB client object configured from the given URL.

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
        :class:`~redis.asyncio.connection.AsyncConnection` when created.
        In the case of conflicting arguments, querystring arguments are used.
        """
        kwargs.update(parse_url(url))
        if kwargs.pop('connection_class', None) is AsyncSSLConnection:
            kwargs['ssl'] = True
        return cls(**kwargs)
    __slots__ = ('_initialize', '_lock', 'cluster_error_retry_attempts', 'command_flags', 'commands_parser', 'connection_error_retry_attempts', 'connection_kwargs', 'encoder', 'node_flags', 'nodes_manager', 'read_from_replicas', 'reinitialize_counter', 'reinitialize_steps', 'response_callbacks', 'result_callbacks')

    def __init__(self, host: Optional[str]=None, port: Union[str, int]=6379, startup_nodes: Optional[List['AsyncClusterNode']]=None, require_full_coverage: bool=True, read_from_replicas: bool=False, reinitialize_steps: int=10, cluster_error_retry_attempts: int=3, connection_error_retry_attempts: int=5, max_connections: int=2 ** 31, db: Union[str, int]=0, path: Optional[str]=None, username: Optional[str]=None, password: Optional[str]=None, client_name: Optional[str]=None, encoding: str='utf-8', encoding_errors: str='strict', decode_responses: bool=False, health_check_interval: float=0, socket_connect_timeout: Optional[float]=None, socket_keepalive: bool=False, socket_keepalive_options: Optional[Mapping[int, Union[int, bytes]]]=None, socket_timeout: Optional[float]=None, ssl: bool=False, ssl_ca_certs: Optional[str]=None, ssl_ca_data: Optional[str]=None, ssl_cert_reqs: str='required', ssl_certfile: Optional[str]=None, ssl_check_hostname: bool=False, ssl_keyfile: Optional[str]=None) -> None:
        if db:
            raise KeyDBClusterException("Argument 'db' must be 0 or None in cluster mode")
        if path:
            raise KeyDBClusterException('Unix domain socket is not supported in cluster mode')
        if (not host or not port) and (not startup_nodes):
            raise KeyDBClusterException('AsyncKeyDBCluster requires at least one node to discover the cluster.\nPlease provide one of the following or use AsyncKeyDBCluster.from_url:\n   - host and port: AsyncKeyDBCluster(host="localhost", port=6379)\n   - startup_nodes: AsyncKeyDBCluster(startup_nodes=[AsyncClusterNode("localhost", 6379), AsyncClusterNode("localhost", 6380)])')
        kwargs: Dict[str, Any] = {'max_connections': max_connections, 'connection_class': AsyncConnection, 'parser_class': ClusterParser, 'username': username, 'password': password, 'client_name': client_name, 'encoding': encoding, 'encoding_errors': encoding_errors, 'decode_responses': decode_responses, 'health_check_interval': health_check_interval, 'socket_connect_timeout': socket_connect_timeout, 'socket_keepalive': socket_keepalive, 'socket_keepalive_options': socket_keepalive_options, 'socket_timeout': socket_timeout}
        if ssl:
            kwargs.update({'connection_class': AsyncSSLConnection, 'ssl_ca_certs': ssl_ca_certs, 'ssl_ca_data': ssl_ca_data, 'ssl_cert_reqs': ssl_cert_reqs, 'ssl_certfile': ssl_certfile, 'ssl_check_hostname': ssl_check_hostname, 'ssl_keyfile': ssl_keyfile})
        if read_from_replicas:
            kwargs['keydb_connect_func'] = self.on_connect
        kwargs['response_callbacks'] = self.__class__.RESPONSE_CALLBACKS.copy()
        self.connection_kwargs = kwargs
        if startup_nodes:
            passed_nodes = []
            for node in startup_nodes:
                passed_nodes.append(AsyncClusterNode(node.host, node.port, **self.connection_kwargs))
            startup_nodes = passed_nodes
        else:
            startup_nodes = []
        if host and port:
            startup_nodes.append(AsyncClusterNode(host, port, **self.connection_kwargs))
        self.nodes_manager = AsyncNodesManager(startup_nodes, require_full_coverage, kwargs)
        self.encoder = Encoder(encoding, encoding_errors, decode_responses)
        self.read_from_replicas = read_from_replicas
        self.reinitialize_steps = reinitialize_steps
        self.cluster_error_retry_attempts = cluster_error_retry_attempts
        self.connection_error_retry_attempts = connection_error_retry_attempts
        self.reinitialize_counter = 0
        self.commands_parser = CommandsParser()
        self.node_flags = self.__class__.NODE_FLAGS.copy()
        self.command_flags = self.__class__.COMMAND_FLAGS.copy()
        self.response_callbacks = kwargs['response_callbacks']
        self.result_callbacks = self.__class__.RESULT_CALLBACKS.copy()
        self.result_callbacks['CLUSTER SLOTS'] = lambda cmd, res, **kwargs: parse_cluster_slots(list(res.values())[0], **kwargs)
        self._initialize = True
        self._lock = asyncio.Lock()

    async def initialize(self) -> 'AsyncKeyDBCluster':
        """Get all nodes from startup nodes & creates connections if not initialized."""
        if self._initialize:
            async with self._lock:
                if self._initialize:
                    try:
                        await self.nodes_manager.initialize()
                        await self.commands_parser.initialize(self.nodes_manager.default_node)
                        self._initialize = False
                    except BaseException:
                        await self.nodes_manager.close()
                        await self.nodes_manager.close('startup_nodes')
                        raise
        return self

    async def close(self) -> None:
        """Close all connections & client if initialized."""
        if not self._initialize:
            async with self._lock:
                if not self._initialize:
                    self._initialize = True
                    await self.nodes_manager.close()
                    await self.nodes_manager.close('startup_nodes')

    async def __aenter__(self) -> 'AsyncKeyDBCluster':
        return await self.initialize()

    async def __aexit__(self, exc_type: None, exc_value: None, traceback: None) -> None:
        await self.close()

    def __await__(self) -> Generator[Any, None, 'AsyncKeyDBCluster']:
        return self.initialize().__await__()
    _DEL_MESSAGE = 'Unclosed AsyncKeyDBCluster client'

    def __del__(self) -> None:
        if hasattr(self, '_initialize') and (not self._initialize):
            warnings.warn(f'{self._DEL_MESSAGE} {self!r}', ResourceWarning, source=self)
            try:
                context = {'client': self, 'message': self._DEL_MESSAGE}
                asyncio.get_running_loop().call_exception_handler(context)
            except RuntimeError:
                ...

    async def on_connect(self, connection: AsyncConnection) -> None:
        await connection.on_connect()
        await connection.send_command('READONLY')
        if str_if_bytes(await connection.read_response_without_lock()) != 'OK':
            raise ConnectionError('READONLY command failed')

    def get_nodes(self) -> List['AsyncClusterNode']:
        """Get all nodes of the cluster."""
        return list(self.nodes_manager.nodes_cache.values())

    def get_primaries(self) -> List['AsyncClusterNode']:
        """Get the primary nodes of the cluster."""
        return self.nodes_manager.get_nodes_by_server_type(PRIMARY)

    def get_replicas(self) -> List['AsyncClusterNode']:
        """Get the replica nodes of the cluster."""
        return self.nodes_manager.get_nodes_by_server_type(REPLICA)

    def get_random_node(self) -> 'AsyncClusterNode':
        """Get a random node of the cluster."""
        return random.choice(list(self.nodes_manager.nodes_cache.values()))

    def get_default_node(self) -> 'AsyncClusterNode':
        """Get the default node of the client."""
        return self.nodes_manager.default_node

    def set_default_node(self, node: 'AsyncClusterNode') -> None:
        """
        Set the default node of the client.

        :raises DataError: if None is passed or node does not exist in cluster.
        """
        if not node or not self.get_node(node_name=node.name):
            raise DataError('The requested node does not exist in the cluster.')
        self.nodes_manager.default_node = node

    def get_node(self, host: Optional[str]=None, port: Optional[int]=None, node_name: Optional[str]=None) -> Optional['AsyncClusterNode']:
        """Get node by (host, port) or node_name."""
        return self.nodes_manager.get_node(host, port, node_name)

    def get_node_from_key(self, key: str, replica: bool=False) -> Optional['AsyncClusterNode']:
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
        """Get the kwargs passed to :class:`~redis.asyncio.connection.AsyncConnection`."""
        return self.connection_kwargs

    def set_response_callback(self, command: str, callback: ResponseCallbackT) -> None:
        """Set a custom response callback."""
        self.response_callbacks[command] = callback

    async def _determine_nodes(self, command: str, *args: Any, node_flag: Optional[str]=None) -> List['AsyncClusterNode']:
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
        if command in ('EVAL', 'EVALSHA'):
            if len(args) < 2:
                raise KeyDBClusterException(f'Invalid args in command: {(command, *args)}')
            keys = args[2:2 + args[1]]
            if not keys:
                return random.randrange(0, REDIS_CLUSTER_HASH_SLOTS)
        else:
            keys = await self.commands_parser.get_keys(command, *args)
            if not keys:
                if command in ('FCALL', 'FCALL_RO'):
                    return random.randrange(0, REDIS_CLUSTER_HASH_SLOTS)
                raise KeyDBClusterException(f'No way to dispatch this command to KeyDB Cluster. Missing key.\nYou can execute the command by specifying target nodes.\nCommand: {args}')
        if len(keys) == 1:
            return self.keyslot(keys[0])
        slots = {self.keyslot(key) for key in keys}
        if len(slots) != 1:
            raise KeyDBClusterException(f'{command} - all keys must map to the same key slot')
        return slots.pop()

    def _is_node_flag(self, target_nodes: Any) -> bool:
        return isinstance(target_nodes, str) and target_nodes in self.node_flags

    def _parse_target_nodes(self, target_nodes: Any) -> List['AsyncClusterNode']:
        if isinstance(target_nodes, list):
            nodes = target_nodes
        elif isinstance(target_nodes, AsyncClusterNode):
            nodes = [target_nodes]
        elif isinstance(target_nodes, dict):
            nodes = list(target_nodes.values())
        else:
            raise TypeError(f'target_nodes type can be one of the following: node_flag (PRIMARIES, REPLICAS, RANDOM, ALL_NODES),AsyncClusterNode, list<AsyncClusterNode>, or dict<any, AsyncClusterNode>. The passed type is {type(target_nodes)}')
        return nodes

    async def execute_command(self, *args: EncodableT, **kwargs: Any) -> Any:
        """
        Execute a raw command on the appropriate cluster node or target_nodes.

        It will retry the command as specified by :attr:`cluster_error_retry_attempts` &
        then raise an exception.

        :param args:
            | Raw command args
        :param kwargs:

            - target_nodes: :attr:`NODE_FLAGS` or :class:`~.AsyncClusterNode`
              or List[:class:`~.AsyncClusterNode`] or Dict[Any, :class:`~.AsyncClusterNode`]
            - Rest of the kwargs are passed to the KeyDB connection

        :raises KeyDBClusterException: if target_nodes is not provided & the command
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
            retry_attempts = 1
        for _ in range(retry_attempts):
            if self._initialize:
                await self.initialize()
            try:
                if not target_nodes_specified:
                    target_nodes = await self._determine_nodes(*args, node_flag=passed_targets)
                    if not target_nodes:
                        raise KeyDBClusterException(f'No targets were found to execute {args} command on')
                if len(target_nodes) == 1:
                    ret = await self._execute_command(target_nodes[0], *args, **kwargs)
                    if command in self.result_callbacks:
                        return self.result_callbacks[command](command, {target_nodes[0].name: ret}, **kwargs)
                    return ret
                else:
                    keys = [node.name for node in target_nodes]
                    values = await asyncio.gather(*(asyncio.ensure_future(self._execute_command(node, *args, **kwargs)) for node in target_nodes))
                    if command in self.result_callbacks:
                        return self.result_callbacks[command](command, dict(zip(keys, values)), **kwargs)
                    return dict(zip(keys, values))
            except Exception as e:
                if type(e) in self.__class__.ERRORS_ALLOW_RETRY:
                    exception = e
                else:
                    raise
        raise exception

    async def _execute_command(self, target_node: 'AsyncClusterNode', *args: Union[KeyT, EncodableT], **kwargs: Any) -> Any:
        asking = moved = False
        redirect_addr = None
        ttl = self.KeyDBClusterRequestTTL
        connection_error_retry_counter = 0
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
            except BusyLoadingError:
                raise
            except (ConnectionError, TimeoutError) as e:
                connection_error_retry_counter += 1
                if connection_error_retry_counter < self.connection_error_retry_attempts:
                    await asyncio.sleep(0.25)
                else:
                    if isinstance(e, MaxConnectionsError):
                        raise
                    await self.close()
                    raise
            except ClusterDownError:
                await self.close()
                await asyncio.sleep(0.25)
                raise
            except MovedError as e:
                self.reinitialize_counter += 1
                if self.reinitialize_steps and self.reinitialize_counter % self.reinitialize_steps == 0:
                    await self.close()
                    self.reinitialize_counter = 0
                else:
                    self.nodes_manager._moved_exception = e
                moved = True
            except AskError as e:
                redirect_addr = get_node_name(host=e.host, port=e.port)
                asking = True
            except TryAgainError:
                if ttl < self.KeyDBClusterRequestTTL / 2:
                    await asyncio.sleep(0.05)
        raise ClusterError('TTL exhausted.')

    def pipeline(self, transaction: Optional[Any]=None, shard_hint: Optional[Any]=None) -> 'AsyncClusterPipeline':
        """
        Create & return a new :class:`~.AsyncClusterPipeline` object.

        Cluster implementation of pipeline does not support transaction or shard_hint.

        :raises KeyDBClusterException: if transaction or shard_hint are truthy values
        """
        if shard_hint:
            raise KeyDBClusterException('shard_hint is deprecated in cluster mode')
        if transaction:
            raise KeyDBClusterException('transaction is deprecated in cluster mode')
        return AsyncClusterPipeline(self)