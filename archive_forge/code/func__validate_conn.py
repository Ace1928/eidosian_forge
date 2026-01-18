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
def _validate_conn(self, conn):
    """
        Called right before a request is made, after the socket is created.
        """
    super(HTTPSConnectionPool, self)._validate_conn(conn)
    if not getattr(conn, 'sock', None):
        conn.connect()
    if not conn.is_verified:
        warnings.warn("Unverified HTTPS request is being made to host '%s'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings" % conn.host, InsecureRequestWarning)
    if getattr(conn, 'proxy_is_verified', None) is False:
        warnings.warn('Unverified HTTPS connection done to an HTTPS proxy. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings', InsecureRequestWarning)