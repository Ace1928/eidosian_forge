import calendar
import collections.abc
import copy
import datetime
import email.utils
from functools import lru_cache
from http.client import responses
import http.cookies
import re
from ssl import SSLError
import time
import unicodedata
from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl
from tornado.escape import native_str, parse_qs_bytes, utf8
from tornado.log import gen_log
from tornado.util import ObjectDict, unicode_type
import typing
from typing import (
class HTTPServerRequest(object):
    """A single HTTP request.

    All attributes are type `str` unless otherwise noted.

    .. attribute:: method

       HTTP request method, e.g. "GET" or "POST"

    .. attribute:: uri

       The requested uri.

    .. attribute:: path

       The path portion of `uri`

    .. attribute:: query

       The query portion of `uri`

    .. attribute:: version

       HTTP version specified in request, e.g. "HTTP/1.1"

    .. attribute:: headers

       `.HTTPHeaders` dictionary-like object for request headers.  Acts like
       a case-insensitive dictionary with additional methods for repeated
       headers.

    .. attribute:: body

       Request body, if present, as a byte string.

    .. attribute:: remote_ip

       Client's IP address as a string.  If ``HTTPServer.xheaders`` is set,
       will pass along the real IP address provided by a load balancer
       in the ``X-Real-Ip`` or ``X-Forwarded-For`` header.

    .. versionchanged:: 3.1
       The list format of ``X-Forwarded-For`` is now supported.

    .. attribute:: protocol

       The protocol used, either "http" or "https".  If ``HTTPServer.xheaders``
       is set, will pass along the protocol used by a load balancer if
       reported via an ``X-Scheme`` header.

    .. attribute:: host

       The requested hostname, usually taken from the ``Host`` header.

    .. attribute:: arguments

       GET/POST arguments are available in the arguments property, which
       maps arguments names to lists of values (to support multiple values
       for individual names). Names are of type `str`, while arguments
       are byte strings.  Note that this is different from
       `.RequestHandler.get_argument`, which returns argument values as
       unicode strings.

    .. attribute:: query_arguments

       Same format as ``arguments``, but contains only arguments extracted
       from the query string.

       .. versionadded:: 3.2

    .. attribute:: body_arguments

       Same format as ``arguments``, but contains only arguments extracted
       from the request body.

       .. versionadded:: 3.2

    .. attribute:: files

       File uploads are available in the files property, which maps file
       names to lists of `.HTTPFile`.

    .. attribute:: connection

       An HTTP request is attached to a single HTTP connection, which can
       be accessed through the "connection" attribute. Since connections
       are typically kept open in HTTP/1.1, multiple requests can be handled
       sequentially on a single connection.

    .. versionchanged:: 4.0
       Moved from ``tornado.httpserver.HTTPRequest``.
    """
    path = None
    query = None
    _body_future = None

    def __init__(self, method: Optional[str]=None, uri: Optional[str]=None, version: str='HTTP/1.0', headers: Optional[HTTPHeaders]=None, body: Optional[bytes]=None, host: Optional[str]=None, files: Optional[Dict[str, List['HTTPFile']]]=None, connection: Optional['HTTPConnection']=None, start_line: Optional['RequestStartLine']=None, server_connection: Optional[object]=None) -> None:
        if start_line is not None:
            method, uri, version = start_line
        self.method = method
        self.uri = uri
        self.version = version
        self.headers = headers or HTTPHeaders()
        self.body = body or b''
        context = getattr(connection, 'context', None)
        self.remote_ip = getattr(context, 'remote_ip', None)
        self.protocol = getattr(context, 'protocol', 'http')
        self.host = host or self.headers.get('Host') or '127.0.0.1'
        self.host_name = split_host_and_port(self.host.lower())[0]
        self.files = files or {}
        self.connection = connection
        self.server_connection = server_connection
        self._start_time = time.time()
        self._finish_time = None
        if uri is not None:
            self.path, sep, self.query = uri.partition('?')
        self.arguments = parse_qs_bytes(self.query, keep_blank_values=True)
        self.query_arguments = copy.deepcopy(self.arguments)
        self.body_arguments = {}

    @property
    def cookies(self) -> Dict[str, http.cookies.Morsel]:
        """A dictionary of ``http.cookies.Morsel`` objects."""
        if not hasattr(self, '_cookies'):
            self._cookies = http.cookies.SimpleCookie()
            if 'Cookie' in self.headers:
                try:
                    parsed = parse_cookie(self.headers['Cookie'])
                except Exception:
                    pass
                else:
                    for k, v in parsed.items():
                        try:
                            self._cookies[k] = v
                        except Exception:
                            pass
        return self._cookies

    def full_url(self) -> str:
        """Reconstructs the full URL for this request."""
        return self.protocol + '://' + self.host + self.uri

    def request_time(self) -> float:
        """Returns the amount of time it took for this request to execute."""
        if self._finish_time is None:
            return time.time() - self._start_time
        else:
            return self._finish_time - self._start_time

    def get_ssl_certificate(self, binary_form: bool=False) -> Union[None, Dict, bytes]:
        """Returns the client's SSL certificate, if any.

        To use client certificates, the HTTPServer's
        `ssl.SSLContext.verify_mode` field must be set, e.g.::

            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain("foo.crt", "foo.key")
            ssl_ctx.load_verify_locations("cacerts.pem")
            ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            server = HTTPServer(app, ssl_options=ssl_ctx)

        By default, the return value is a dictionary (or None, if no
        client certificate is present).  If ``binary_form`` is true, a
        DER-encoded form of the certificate is returned instead.  See
        SSLSocket.getpeercert() in the standard library for more
        details.
        http://docs.python.org/library/ssl.html#sslsocket-objects
        """
        try:
            if self.connection is None:
                return None
            return self.connection.stream.socket.getpeercert(binary_form=binary_form)
        except SSLError:
            return None

    def _parse_body(self) -> None:
        parse_body_arguments(self.headers.get('Content-Type', ''), self.body, self.body_arguments, self.files, self.headers)
        for k, v in self.body_arguments.items():
            self.arguments.setdefault(k, []).extend(v)

    def __repr__(self) -> str:
        attrs = ('protocol', 'host', 'method', 'uri', 'version', 'remote_ip')
        args = ', '.join(['%s=%r' % (n, getattr(self, n)) for n in attrs])
        return '%s(%s)' % (self.__class__.__name__, args)