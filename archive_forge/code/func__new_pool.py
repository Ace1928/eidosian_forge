from __future__ import absolute_import
import collections
import functools
import logging
from ._collections import RecentlyUsedContainer
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, port_by_scheme
from .exceptions import (
from .packages import six
from .packages.six.moves.urllib.parse import urljoin
from .request import RequestMethods
from .util.proxy import connection_requires_http_tunnel
from .util.retry import Retry
from .util.url import parse_url
def _new_pool(self, scheme, host, port, request_context=None):
    """
        Create a new :class:`urllib3.connectionpool.ConnectionPool` based on host, port, scheme, and
        any additional pool keyword arguments.

        If ``request_context`` is provided, it is provided as keyword arguments
        to the pool class used. This method is used to actually create the
        connection pools handed out by :meth:`connection_from_url` and
        companion methods. It is intended to be overridden for customization.
        """
    pool_cls = self.pool_classes_by_scheme[scheme]
    if request_context is None:
        request_context = self.connection_pool_kw.copy()
    for key in ('scheme', 'host', 'port'):
        request_context.pop(key, None)
    if scheme == 'http':
        for kw in SSL_KEYWORDS:
            request_context.pop(kw, None)
    return pool_cls(host, port, **request_context)