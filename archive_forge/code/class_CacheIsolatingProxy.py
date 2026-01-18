import copy
from dogpile.cache import proxy
from oslo_cache import core as cache
class CacheIsolatingProxy(proxy.ProxyBackend):
    """Proxy that forces a memory copy of stored values.

    The default in-memory cache-region does not perform a copy on values it
    is meant to cache.  Therefore if the value is modified after set or after
    get, the cached value also is modified.  This proxy does a copy as the last
    thing before storing data.

    In your application's tests, you'll want to set this as a proxy for the
    in-memory cache, like this::

        self.config_fixture.config(
            group='cache',
            backend='dogpile.cache.memory',
            enabled=True,
            proxies=['oslo_cache.testing.CacheIsolatingProxy'])

    """

    def get(self, key):
        return _copy_value(self.proxied.get(key))

    def set(self, key, value):
        self.proxied.set(key, _copy_value(value))