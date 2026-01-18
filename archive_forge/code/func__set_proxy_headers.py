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
def _set_proxy_headers(self, url, headers=None):
    """
        Sets headers needed by proxies: specifically, the Accept and Host
        headers. Only sets headers not provided by the user.
        """
    headers_ = {'Accept': '*/*'}
    netloc = parse_url(url).netloc
    if netloc:
        headers_['Host'] = netloc
    if headers:
        headers_.update(headers)
    return headers_