from dogpile.cache import api
from oslo_cache import core
from oslo_serialization import jsonutils
class Etcd3gwCacheBackend(api.CacheBackend):
    DEFAULT_TIMEOUT = 30
    DEFAULT_HOST = 'localhost'
    DEFAULT_PORT = 2379

    def __init__(self, arguments):
        self.host = arguments.get('host', self.DEFAULT_HOST)
        self.port = arguments.get('port', self.DEFAULT_PORT)
        self.timeout = int(arguments.get('timeout', self.DEFAULT_TIMEOUT))
        import etcd3gw
        self._client = etcd3gw.client(host=self.host, port=self.port, timeout=self.timeout)

    def get(self, key):
        values = self._client.get(key, False)
        if not values:
            return core.NO_VALUE
        value, metadata = jsonutils.loads(values[0])
        return api.CachedValue(value, metadata)

    def get_multi(self, keys):
        """Retrieves the value for a list of keys."""
        return [self.get(key) for key in keys]

    def set(self, key, value):
        self.set_multi({key: value})

    def set_multi(self, mapping):
        lease = None
        if self.timeout:
            lease = self._client.lease(ttl=self.timeout)
        for key, value in mapping.items():
            self._client.put(key, jsonutils.dumps(value), lease)

    def delete(self, key):
        self._client.delete(key)

    def delete_multi(self, keys):
        for key in keys:
            self._client.delete(key)