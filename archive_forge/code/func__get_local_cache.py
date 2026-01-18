from dogpile.cache import api
from dogpile.cache import proxy
from oslo_context import context as oslo_context
from oslo_serialization import msgpackutils
def _get_local_cache(self, key):
    ctx = self._get_request_context()
    try:
        value = getattr(ctx, self._get_request_key(key))
    except AttributeError:
        return api.NO_VALUE
    value = msgpackutils.loads(value)
    return api.CachedValue(payload=value['payload'], metadata=value['metadata'])