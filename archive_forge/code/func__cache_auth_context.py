import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
def _cache_auth_context(self, context):
    """
        Store an authentication context in memory and the cache.

        :param context: Authentication context to cache.
        :type key: :class:`.OpenStackAuthenticationContext`
        """
    self.urls = context.urls
    self.auth_token = context.token
    self.auth_token_expires = context.expiration
    self.auth_user_info = context.user
    self.auth_user_roles = context.roles
    if self.auth_cache is not None:
        self.auth_cache.put(self._cache_key, context)