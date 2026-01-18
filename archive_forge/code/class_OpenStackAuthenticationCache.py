import datetime
from collections import namedtuple
from libcloud.utils.py3 import httplib
from libcloud.common.base import Response, ConnectionUserAndKey, CertificateConnection
from libcloud.compute.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.utils.iso8601 import parse_date
class OpenStackAuthenticationCache:
    """
    Base class for external OpenStack authentication caches.

    Authentication tokens are always cached in memory in
    :class:`OpenStackIdentityConnection`.auth_token and related fields.  These
    tokens are lost when the driver is garbage collected.  To share tokens
    among multiple drivers, processes, or systems, use an
    :class:`OpenStackAuthenticationCache` in
    OpenStackIdentityConnection.auth_cache.

    Cache implementors should inherit this class and define the methods below.
    """

    def get(self, key):
        """
        Get an authentication context from the cache.

        :param key: Key to fetch.
        :type key: :class:`.OpenStackAuthenticationCacheKey`

        :return: The cached context for the given key, if present; None if not.
        :rtype: :class:`OpenStackAuthenticationContext`
        """
        raise NotImplementedError

    def put(self, key, context):
        """
        Put an authentication context into the cache.

        :param key: Key where the context will be stored.
        :type key: :class:`.OpenStackAuthenticationCacheKey`

        :param context: The context to cache.
        :type context: :class:`.OpenStackAuthenticationContext`
        """
        raise NotImplementedError

    def clear(self, key):
        """
        Clear an authentication context from the cache.

        :param key: Key to clear.
        :type key: :class:`.OpenStackAuthenticationCacheKey`
        """
        raise NotImplementedError