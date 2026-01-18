import contextlib
import hashlib
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.i18n import _
class _FakeClient(object):
    """Replicates a tiny subset of memcached client interface."""

    def __init__(self, *args, **kwargs):
        self.cache = {}

    def get(self, key):
        """Retrieve the value for a key or None.

        This expunges expired keys during each get.
        """
        now = timeutils.utcnow_ts()
        for k in list(self.cache):
            timeout, _value = self.cache[k]
            if timeout and now >= timeout:
                del self.cache[k]
        return self.cache.get(key, (0, None))[1]

    def set(self, key, value, time=0, min_compress_len=0):
        """Set the value for a key."""
        timeout = 0
        if time != 0:
            timeout = timeutils.utcnow_ts() + time
        self.cache[key] = (timeout, value)
        return True

    def add(self, key, value, time=0, min_compress_len=0):
        """Set the value for a key if it doesn't exist."""
        if self.get(key) is not None:
            return False
        return self.set(key, value, time, min_compress_len)

    def incr(self, key, delta=1):
        """Increment the value for a key."""
        value = self.get(key)
        if value is None:
            return None
        new_value = int(value) + delta
        self.cache[key] = (self.cache[key][0], str(new_value))
        return new_value

    def delete(self, key, time=0):
        """Delete the value associated with a key."""
        if key in self.cache:
            del self.cache[key]