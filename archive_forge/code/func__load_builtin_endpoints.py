import os
import boto
from boto.compat import json
from boto.exception import BotoClientError
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def _load_builtin_endpoints(_cache=_endpoints_cache):
    """Loads the builtin endpoints in the legacy format."""
    if _cache:
        return _cache
    endpoints = _load_json_file(boto.ENDPOINTS_PATH)
    resolver = BotoEndpointResolver(endpoints)
    builder = StaticEndpointBuilder(resolver)
    endpoints = builder.build_static_endpoints()
    _cache.update(endpoints)
    return _cache