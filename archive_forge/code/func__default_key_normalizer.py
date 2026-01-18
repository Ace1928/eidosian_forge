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
def _default_key_normalizer(key_class, request_context):
    """
    Create a pool key out of a request context dictionary.

    According to RFC 3986, both the scheme and host are case-insensitive.
    Therefore, this function normalizes both before constructing the pool
    key for an HTTPS request. If you wish to change this behaviour, provide
    alternate callables to ``key_fn_by_scheme``.

    :param key_class:
        The class to use when constructing the key. This should be a namedtuple
        with the ``scheme`` and ``host`` keys at a minimum.
    :type  key_class: namedtuple
    :param request_context:
        A dictionary-like object that contain the context for a request.
    :type  request_context: dict

    :return: A namedtuple that can be used as a connection pool key.
    :rtype:  PoolKey
    """
    context = request_context.copy()
    context['scheme'] = context['scheme'].lower()
    context['host'] = context['host'].lower()
    for key in ('headers', '_proxy_headers', '_socks_options'):
        if key in context and context[key] is not None:
            context[key] = frozenset(context[key].items())
    socket_opts = context.get('socket_options')
    if socket_opts is not None:
        context['socket_options'] = tuple(socket_opts)
    for key in list(context.keys()):
        context['key_' + key] = context.pop(key)
    for field in key_class._fields:
        if field not in context:
            context[field] = None
    return key_class(**context)