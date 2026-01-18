import contextlib
import hashlib
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.i18n import _
class _CachePool(list):
    """A lazy pool of cache references."""

    def __init__(self, memcached_servers, log):
        self._memcached_servers = memcached_servers
        if not self._memcached_servers:
            log.warning("Using the in-process token cache is deprecated as of the 4.2.0 release and may be removed in the 5.0.0 release or the 'O' development cycle. The in-process cache causes inconsistent results and high memory usage. When the feature is removed the auth_token middleware will not cache tokens by default which may result in performance issues. It is recommended to use  memcache for the auth_token token cache by setting the memcached_servers option.")

    @contextlib.contextmanager
    def reserve(self):
        """Context manager to manage a pooled cache reference."""
        try:
            c = self.pop()
        except IndexError:
            if self._memcached_servers:
                import memcache
                c = memcache.Client(self._memcached_servers, debug=0)
            else:
                c = _FakeClient()
        try:
            yield c
        finally:
            self.append(c)