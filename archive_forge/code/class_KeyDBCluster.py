import typing
import threading
from redis.commands import CommandsParser
from redis.cluster import (
from redis.asyncio.cluster import (
from redis.retry import Retry
from redis.exceptions import (
from aiokeydb.v2.backoff import default_backoff
from aiokeydb.v2.connection import (
from aiokeydb.v2.core import (
class KeyDBCluster(RedisCluster):

    @classmethod
    def from_url(cls, url, **kwargs):
        """
        Return a Redis client object configured from the given URL

        For example::

            redis://[[username]:[password]]@localhost:6379/0
            rediss://[[username]:[password]]@localhost:6379/0
            unix://[username@]/path/to/socket.sock?db=0[&password=password]

        Three URL schemes are supported:

        - `redis://` creates a TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/redis>
        - `rediss://` creates a SSL wrapped TCP socket connection. See more at:
          <https://www.iana.org/assignments/uri-schemes/prov/rediss>
        - ``unix://``: creates a Unix Domain Socket connection.

        The username, password, hostname, path and all querystring values
        are passed through urllib.parse.unquote in order to replace any
        percent-encoded values with their corresponding characters.

        There are several ways to specify a database number. The first value
        found will be used:

            1. A ``db`` querystring option, e.g. redis://localhost?db=0
            2. If using the redis:// or rediss:// schemes, the path argument
               of the url, e.g. redis://localhost/0
            3. A ``db`` keyword argument to this function.

        If none of these options are specified, the default db=0 is used.

        All querystring options are cast to their appropriate Python types.
        Boolean arguments can be specified with string values "True"/"False"
        or "Yes"/"No". Values that cannot be properly cast cause a
        ``ValueError`` to be raised. Once parsed, the querystring arguments
        and keyword arguments are passed to the ``ConnectionPool``'s
        class initializer. In the case of conflicting arguments, querystring
        arguments always win.

        """
        return cls(url=url, **kwargs)

    def __init__(self, host: typing.Optional[str]=None, port: int=6379, startup_nodes: typing.Optional[typing.List['ClusterNode']]=None, cluster_error_retry_attempts: int=3, retry: typing.Optional['Retry']=None, require_full_coverage: bool=False, reinitialize_steps: int=5, read_from_replicas: bool=False, dynamic_startup_nodes: bool=True, url: typing.Optional[str]=None, **kwargs):
        """
         Initialize a new RedisCluster client.

         :param startup_nodes:
             List of nodes from which initial bootstrapping can be done
         :param host:
             Can be used to point to a startup node
         :param port:
             Can be used to point to a startup node
         :param require_full_coverage:
            When set to False (default value): the client will not require a
            full coverage of the slots. However, if not all slots are covered,
            and at least one node has 'cluster-require-full-coverage' set to
            'yes,' the server will throw a ClusterDownError for some key-based
            commands. See -
            https://redis.io/topics/cluster-tutorial#redis-cluster-configuration-parameters
            When set to True: all slots must be covered to construct the
            cluster client. If not all slots are covered, RedisClusterException
            will be thrown.
        :param read_from_replicas:
             Enable read from replicas in READONLY mode. You can read possibly
             stale data.
             When set to true, read commands will be assigned between the
             primary and its replications in a Round-Robin manner.
         :param dynamic_startup_nodes:
             Set the RedisCluster's startup nodes to all of the discovered nodes.
             If true (default value), the cluster's discovered nodes will be used to
             determine the cluster nodes-slots mapping in the next topology refresh.
             It will remove the initial passed startup nodes if their endpoints aren't
             listed in the CLUSTER SLOTS output.
             If you use dynamic DNS endpoints for startup nodes but CLUSTER SLOTS lists
             specific IP addresses, it is best to set it to false.
        :param cluster_error_retry_attempts:
             Number of times to retry before raising an error when
             :class:`~.TimeoutError` or :class:`~.ConnectionError` or
             :class:`~.ClusterDownError` are encountered
        :param reinitialize_steps:
            Specifies the number of MOVED errors that need to occur before
            reinitializing the whole cluster topology. If a MOVED error occurs
            and the cluster does not need to be reinitialized on this current
            error handling, only the MOVED slot will be patched with the
            redirected node.
            To reinitialize the cluster on every MOVED error, set
            reinitialize_steps to 1.
            To avoid reinitializing the cluster on moved errors, set
            reinitialize_steps to 0.

         :**kwargs:
             Extra arguments that will be sent into Redis instance when created
             (See Official redis-py doc for supported kwargs
         [https://github.com/andymccurdy/redis-py/blob/master/redis/client.py])
             Some kwargs are not supported and will raise a
             RedisClusterException:
                 - db (Redis do not support database SELECT in cluster mode)
        """
        if startup_nodes is None:
            startup_nodes = []
        if 'db' in kwargs:
            raise RedisClusterException("Argument 'db' is not possible to use in cluster mode")
        from_url = False
        if url is not None:
            from_url = True
            url_options = parse_url(url)
            if 'path' in url_options:
                raise RedisClusterException('RedisCluster does not currently support Unix Domain Socket connections')
            if 'db' in url_options and url_options['db'] != 0:
                raise RedisClusterException('A ``db`` querystring option can only be 0 in cluster mode')
            kwargs.update(url_options)
            host = kwargs.get('host')
            port = kwargs.get('port', port)
            startup_nodes.append(ClusterNode(host, port))
        elif host is not None and port is not None:
            startup_nodes.append(ClusterNode(host, port))
        elif len(startup_nodes) == 0:
            raise RedisClusterException("RedisCluster requires at least one node to discover the cluster. Please provide one of the followings:\n1. host and port, for example:\n RedisCluster(host='localhost', port=6379)\n2. list of startup nodes, for example:\n RedisCluster(startup_nodes=[ClusterNode('localhost', 6379), ClusterNode('localhost', 6378)])")
        self.user_on_connect_func = kwargs.pop('redis_connect_func', None)
        kwargs['redis_connect_func'] = self.on_connect
        kwargs = cleanup_kwargs(**kwargs)
        if retry:
            self.retry = retry
            kwargs.update({'retry': self.retry})
        else:
            kwargs.update({'retry': Retry(default_backoff(), 0)})
        self.encoder = Encoder(kwargs.get('encoding', 'utf-8'), kwargs.get('encoding_errors', 'strict'), kwargs.get('decode_responses', False))
        self.cluster_error_retry_attempts = cluster_error_retry_attempts
        self.command_flags = self.__class__.COMMAND_FLAGS.copy()
        self.node_flags = self.__class__.NODE_FLAGS.copy()
        self.read_from_replicas = read_from_replicas
        self.reinitialize_counter = 0
        self.reinitialize_steps = reinitialize_steps
        self.nodes_manager = NodesManager(startup_nodes=startup_nodes, from_url=from_url, require_full_coverage=require_full_coverage, dynamic_startup_nodes=dynamic_startup_nodes, **kwargs)
        self.cluster_response_callbacks = CaseInsensitiveDict(self.__class__.CLUSTER_COMMANDS_RESPONSE_CALLBACKS)
        self.result_callbacks = CaseInsensitiveDict(self.__class__.RESULT_CALLBACKS)
        self.commands_parser = CommandsParser(self)
        self._lock = threading.Lock()