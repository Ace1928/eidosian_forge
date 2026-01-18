import functools
from dogpile.cache.backends import memcached as memcached_backend
from oslo_cache import _memcache_pool
from oslo_cache import exception
class PooledMemcachedBackend(memcached_backend.MemcachedBackend):
    """Memcached backend that does connection pooling.

    This memcached backend only allows for reuse of a client object,
    prevents too many client object from being instantiated, and maintains
    proper tracking of dead servers so as to limit delays when a server
    (or all servers) become unavailable.

    This backend doesn't allow to load balance things between servers.

    Memcached isn't HA. Values aren't automatically replicated between servers
    unless the client went out and wrote the value multiple time.

    The memcache server to use is determined by `python-memcached` itself by
    picking the host to use (from the given server list) based on a key hash.
    """

    def __init__(self, arguments):
        super(PooledMemcachedBackend, self).__init__(arguments)
        if arguments.get('sasl_enabled', False):
            if arguments.get('username') is None or arguments.get('password') is None:
                raise exception.ConfigurationError('username and password should be configured to use SASL authentication.')
            if not _bmemcache_pool:
                raise ImportError('python-binary-memcached package is missing')
            self.client_pool = _bmemcache_pool.BMemcacheClientPool(self.url, arguments, maxsize=arguments.get('pool_maxsize', 10), unused_timeout=arguments.get('pool_unused_timeout', 60), conn_get_timeout=arguments.get('pool_connection_get_timeout', 10))
        else:
            self.client_pool = _memcache_pool.MemcacheClientPool(self.url, arguments, maxsize=arguments.get('pool_maxsize', 10), unused_timeout=arguments.get('pool_unused_timeout', 60), conn_get_timeout=arguments.get('pool_connection_get_timeout', 10))

    @property
    def client(self):
        return ClientProxy(self.client_pool)