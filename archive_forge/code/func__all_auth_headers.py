import itertools
from oslo_serialization import jsonutils
import webob
def _all_auth_headers(self):
    """All the authentication headers that can be set on the request."""
    yield self._SERVICE_CATALOG_HEADER
    yield self._USER_STATUS_HEADER
    yield self._SERVICE_STATUS_HEADER
    yield self._ADMIN_PROJECT_HEADER
    for header in self._DEPRECATED_HEADER_MAP:
        yield header
    prefixes = (self._USER_HEADER_PREFIX, self._SERVICE_HEADER_PREFIX)
    for tmpl, prefix in itertools.product(self._HEADER_TEMPLATE, prefixes):
        yield (tmpl % prefix)
    for prefix in prefixes:
        yield (self._ROLES_TEMPLATE % prefix)