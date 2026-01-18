import pickle
import random
import re
from django.core.cache.backends.base import DEFAULT_TIMEOUT, BaseCache
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
class RedisCacheClient:

    def __init__(self, servers, serializer=None, pool_class=None, parser_class=None, **options):
        import redis
        self._lib = redis
        self._servers = servers
        self._pools = {}
        self._client = self._lib.Redis
        if isinstance(pool_class, str):
            pool_class = import_string(pool_class)
        self._pool_class = pool_class or self._lib.ConnectionPool
        if isinstance(serializer, str):
            serializer = import_string(serializer)
        if callable(serializer):
            serializer = serializer()
        self._serializer = serializer or RedisSerializer()
        if isinstance(parser_class, str):
            parser_class = import_string(parser_class)
        parser_class = parser_class or self._lib.connection.DefaultParser
        self._pool_options = {'parser_class': parser_class, **options}

    def _get_connection_pool_index(self, write):
        if write or len(self._servers) == 1:
            return 0
        return random.randint(1, len(self._servers) - 1)

    def _get_connection_pool(self, write):
        index = self._get_connection_pool_index(write)
        if index not in self._pools:
            self._pools[index] = self._pool_class.from_url(self._servers[index], **self._pool_options)
        return self._pools[index]

    def get_client(self, key=None, *, write=False):
        pool = self._get_connection_pool(write)
        return self._client(connection_pool=pool)

    def add(self, key, value, timeout):
        client = self.get_client(key, write=True)
        value = self._serializer.dumps(value)
        if timeout == 0:
            if (ret := bool(client.set(key, value, nx=True))):
                client.delete(key)
            return ret
        else:
            return bool(client.set(key, value, ex=timeout, nx=True))

    def get(self, key, default):
        client = self.get_client(key)
        value = client.get(key)
        return default if value is None else self._serializer.loads(value)

    def set(self, key, value, timeout):
        client = self.get_client(key, write=True)
        value = self._serializer.dumps(value)
        if timeout == 0:
            client.delete(key)
        else:
            client.set(key, value, ex=timeout)

    def touch(self, key, timeout):
        client = self.get_client(key, write=True)
        if timeout is None:
            return bool(client.persist(key))
        else:
            return bool(client.expire(key, timeout))

    def delete(self, key):
        client = self.get_client(key, write=True)
        return bool(client.delete(key))

    def get_many(self, keys):
        client = self.get_client(None)
        ret = client.mget(keys)
        return {k: self._serializer.loads(v) for k, v in zip(keys, ret) if v is not None}

    def has_key(self, key):
        client = self.get_client(key)
        return bool(client.exists(key))

    def incr(self, key, delta):
        client = self.get_client(key, write=True)
        if not client.exists(key):
            raise ValueError("Key '%s' not found." % key)
        return client.incr(key, delta)

    def set_many(self, data, timeout):
        client = self.get_client(None, write=True)
        pipeline = client.pipeline()
        pipeline.mset({k: self._serializer.dumps(v) for k, v in data.items()})
        if timeout is not None:
            for key in data:
                pipeline.expire(key, timeout)
        pipeline.execute()

    def delete_many(self, keys):
        client = self.get_client(None, write=True)
        client.delete(*keys)

    def clear(self):
        client = self.get_client(None, write=True)
        return bool(client.flushdb())