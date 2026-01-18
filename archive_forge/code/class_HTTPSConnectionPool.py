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
class HTTPSConnectionPool(HTTPConnectionPool):
    """
    Same as :class:`.HTTPConnectionPool`, but HTTPS.

    :class:`.HTTPSConnection` uses one of ``assert_fingerprint``,
    ``assert_hostname`` and ``host`` in this order to verify connections.
    If ``assert_hostname`` is False, no verification is done.

    The ``key_file``, ``cert_file``, ``cert_reqs``, ``ca_certs``,
    ``ca_cert_dir``, ``ssl_version``, ``key_password`` are only used if :mod:`ssl`
    is available and are fed into :meth:`urllib3.util.ssl_wrap_socket` to upgrade
    the connection socket into an SSL socket.
    """
    scheme = 'https'
    ConnectionCls = HTTPSConnection

    def __init__(self, host, port=None, strict=False, timeout=Timeout.DEFAULT_TIMEOUT, maxsize=1, block=False, headers=None, retries=None, _proxy=None, _proxy_headers=None, key_file=None, cert_file=None, cert_reqs=None, key_password=None, ca_certs=None, ssl_version=None, assert_hostname=None, assert_fingerprint=None, ca_cert_dir=None, **conn_kw):
        HTTPConnectionPool.__init__(self, host, port, strict, timeout, maxsize, block, headers, retries, _proxy, _proxy_headers, **conn_kw)
        self.key_file = key_file
        self.cert_file = cert_file
        self.cert_reqs = cert_reqs
        self.key_password = key_password
        self.ca_certs = ca_certs
        self.ca_cert_dir = ca_cert_dir
        self.ssl_version = ssl_version
        self.assert_hostname = assert_hostname
        self.assert_fingerprint = assert_fingerprint

    def _prepare_conn(self, conn):
        """
        Prepare the ``connection`` for :meth:`urllib3.util.ssl_wrap_socket`
        and establish the tunnel if proxy is used.
        """
        if isinstance(conn, VerifiedHTTPSConnection):
            conn.set_cert(key_file=self.key_file, key_password=self.key_password, cert_file=self.cert_file, cert_reqs=self.cert_reqs, ca_certs=self.ca_certs, ca_cert_dir=self.ca_cert_dir, assert_hostname=self.assert_hostname, assert_fingerprint=self.assert_fingerprint)
            conn.ssl_version = self.ssl_version
        return conn

    def _prepare_proxy(self, conn):
        """
        Establishes a tunnel connection through HTTP CONNECT.

        Tunnel connection is established early because otherwise httplib would
        improperly set Host: header to proxy's IP:port.
        """
        conn.set_tunnel(self._proxy_host, self.port, self.proxy_headers)
        if self.proxy.scheme == 'https':
            conn.tls_in_tls_required = True
        conn.connect()

    def _new_conn(self):
        """
        Return a fresh :class:`http.client.HTTPSConnection`.
        """
        self.num_connections += 1
        log.debug('Starting new HTTPS connection (%d): %s:%s', self.num_connections, self.host, self.port or '443')
        if not self.ConnectionCls or self.ConnectionCls is DummyConnection:
            raise SSLError("Can't connect to HTTPS URL because the SSL module is not available.")
        actual_host = self.host
        actual_port = self.port
        if self.proxy is not None:
            actual_host = self.proxy.host
            actual_port = self.proxy.port
        conn = self.ConnectionCls(host=actual_host, port=actual_port, timeout=self.timeout.connect_timeout, strict=self.strict, cert_file=self.cert_file, key_file=self.key_file, key_password=self.key_password, **self.conn_kw)
        return self._prepare_conn(conn)

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