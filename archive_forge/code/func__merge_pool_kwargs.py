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
def _merge_pool_kwargs(self, override):
    """
        Merge a dictionary of override values for self.connection_pool_kw.

        This does not modify self.connection_pool_kw and returns a new dict.
        Any keys in the override dictionary with a value of ``None`` are
        removed from the merged dictionary.
        """
    base_pool_kwargs = self.connection_pool_kw.copy()
    if override:
        for key, value in override.items():
            if value is None:
                try:
                    del base_pool_kwargs[key]
                except KeyError:
                    pass
            else:
                base_pool_kwargs[key] = value
    return base_pool_kwargs