import contextlib
import hashlib
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.i18n import _
def _get_cache_key(self, token_id):
    context = memcache_crypt.derive_keys(token_id, self._secret_key, self._security_strategy)
    key = self._CACHE_KEY_TEMPLATE % memcache_crypt.get_cache_key(context)
    return (key, context)