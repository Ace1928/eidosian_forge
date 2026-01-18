from __future__ import absolute_import
import errno
import logging
import re
import socket
import sys
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout
from .connection import (
from .exceptions import (
from .packages import six
from .packages.six.moves import queue
from .request import RequestMethods
from .response import HTTPResponse
from .util.connection import is_connection_dropped
from .util.proxy import connection_requires_http_tunnel
from .util.queue import LifoQueue
from .util.request import set_file_position
from .util.response import assert_header_parsing
from .util.retry import Retry
from .util.ssl_match_hostname import CertificateError
from .util.timeout import Timeout
from .util.url import Url, _encode_target
from .util.url import _normalize_host as normalize_host
from .util.url import get_host, parse_url
def _get_conn(self, timeout=None):
    """
        Get a connection. Will return a pooled connection if one is available.

        If no connections are available and :prop:`.block` is ``False``, then a
        fresh connection is returned.

        :param timeout:
            Seconds to wait before giving up and raising
            :class:`urllib3.exceptions.EmptyPoolError` if the pool is empty and
            :prop:`.block` is ``True``.
        """
    conn = None
    try:
        conn = self.pool.get(block=self.block, timeout=timeout)
    except AttributeError:
        raise ClosedPoolError(self, 'Pool is closed.')
    except queue.Empty:
        if self.block:
            raise EmptyPoolError(self, 'Pool reached maximum size and no more connections are allowed.')
        pass
    if conn and is_connection_dropped(conn):
        log.debug('Resetting dropped connection: %s', self.host)
        conn.close()
        if getattr(conn, 'auto_open', 1) == 0:
            conn = None
    return conn or self._new_conn()