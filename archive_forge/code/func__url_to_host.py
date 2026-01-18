import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import partial
from ipaddress import ip_address
import itertools
import logging
import random
import ssl
import struct
import urllib.parse
from typing import List, Optional, Union
import trio
import trio.abc
from wsproto import ConnectionType, WSConnection
from wsproto.connection import ConnectionState
import wsproto.frame_protocol as wsframeproto
from wsproto.events import (
import wsproto.utilities
def _url_to_host(url, ssl_context):
    """
    Convert a WebSocket URL to a (host,port,resource) tuple.

    The returned ``ssl_context`` is either the same object that was passed in,
    or if ``ssl_context`` is None, then a bool indicating if a default SSL
    context needs to be created.

    :param str url: A WebSocket URL.
    :type ssl_context: ssl.SSLContext or None
    :returns: A tuple of ``(host, port, resource, ssl_context)``.
    """
    url = str(url)
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ('ws', 'wss'):
        raise ValueError('WebSocket URL scheme must be "ws:" or "wss:"')
    if ssl_context is None:
        ssl_context = parts.scheme == 'wss'
    elif parts.scheme == 'ws':
        raise ValueError('SSL context must be None for ws: URL scheme')
    host = parts.hostname
    if parts.port is not None:
        port = parts.port
    else:
        port = 443 if ssl_context else 80
    path_qs = parts.path
    if not path_qs:
        path_qs = '/'
    if '?' in url:
        path_qs += '?' + parts.query
    return (host, port, path_qs, ssl_context)