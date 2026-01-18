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
def _validate_proxy_scheme_url_selection(self, url_scheme):
    """
        Validates that were not attempting to do TLS in TLS connections on
        Python2 or with unsupported SSL implementations.
        """
    if self.proxy is None or url_scheme != 'https':
        return
    if self.proxy.scheme != 'https':
        return
    if six.PY2 and (not self.proxy_config.use_forwarding_for_https):
        raise ProxySchemeUnsupported("Contacting HTTPS destinations through HTTPS proxies 'via CONNECT tunnels' is not supported in Python 2")