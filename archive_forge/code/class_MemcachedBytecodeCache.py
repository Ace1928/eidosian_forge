import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
class MemcachedBytecodeCache(BytecodeCache):
    """This class implements a bytecode cache that uses a memcache cache for
    storing the information.  It does not enforce a specific memcache library
    (tummy's memcache or cmemcache) but will accept any class that provides
    the minimal interface required.

    Libraries compatible with this class:

    -   `cachelib <https://github.com/pallets/cachelib>`_
    -   `python-memcached <https://pypi.org/project/python-memcached/>`_

    (Unfortunately the django cache interface is not compatible because it
    does not support storing binary data, only text. You can however pass
    the underlying cache client to the bytecode cache which is available
    as `django.core.cache.cache._client`.)

    The minimal interface for the client passed to the constructor is this:

    .. class:: MinimalClientInterface

        .. method:: set(key, value[, timeout])

            Stores the bytecode in the cache.  `value` is a string and
            `timeout` the timeout of the key.  If timeout is not provided
            a default timeout or no timeout should be assumed, if it's
            provided it's an integer with the number of seconds the cache
            item should exist.

        .. method:: get(key)

            Returns the value for the cache key.  If the item does not
            exist in the cache the return value must be `None`.

    The other arguments to the constructor are the prefix for all keys that
    is added before the actual cache key and the timeout for the bytecode in
    the cache system.  We recommend a high (or no) timeout.

    This bytecode cache does not support clearing of used items in the cache.
    The clear method is a no-operation function.

    .. versionadded:: 2.7
       Added support for ignoring memcache errors through the
       `ignore_memcache_errors` parameter.
    """

    def __init__(self, client: '_MemcachedClient', prefix: str='jinja2/bytecode/', timeout: t.Optional[int]=None, ignore_memcache_errors: bool=True):
        self.client = client
        self.prefix = prefix
        self.timeout = timeout
        self.ignore_memcache_errors = ignore_memcache_errors

    def load_bytecode(self, bucket: Bucket) -> None:
        try:
            code = self.client.get(self.prefix + bucket.key)
        except Exception:
            if not self.ignore_memcache_errors:
                raise
        else:
            bucket.bytecode_from_string(code)

    def dump_bytecode(self, bucket: Bucket) -> None:
        key = self.prefix + bucket.key
        value = bucket.bytecode_to_string()
        try:
            if self.timeout is not None:
                self.client.set(key, value, self.timeout)
            else:
                self.client.set(key, value)
        except Exception:
            if not self.ignore_memcache_errors:
                raise