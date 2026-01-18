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
def _prepare_conn(self, conn):
    """
        Prepare the ``connection`` for :meth:`urllib3.util.ssl_wrap_socket`
        and establish the tunnel if proxy is used.
        """
    if isinstance(conn, VerifiedHTTPSConnection):
        conn.set_cert(key_file=self.key_file, key_password=self.key_password, cert_file=self.cert_file, cert_reqs=self.cert_reqs, ca_certs=self.ca_certs, ca_cert_dir=self.ca_cert_dir, assert_hostname=self.assert_hostname, assert_fingerprint=self.assert_fingerprint)
        conn.ssl_version = self.ssl_version
    return conn