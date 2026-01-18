import contextlib
import hashlib
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.i18n import _
class _EnvCachePool(object):
    """A cache pool that has been passed through ENV variables."""

    def __init__(self, cache):
        self._environment_cache = cache

    @contextlib.contextmanager
    def reserve(self):
        """Context manager to manage a pooled cache reference."""
        yield self._environment_cache