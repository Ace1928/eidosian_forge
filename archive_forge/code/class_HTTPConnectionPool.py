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
class HTTPConnectionPool(ConnectionPool, RequestMethods):
    """
    Thread-safe connection pool for one host.

    :param host:
        Host used for this HTTP Connection (e.g. "localhost"), passed into
        :class:`http.client.HTTPConnection`.

    :param port:
        Port used for this HTTP Connection (None is equivalent to 80), passed
        into :class:`http.client.HTTPConnection`.

    :param strict:
        Causes BadStatusLine to be raised if the status line can't be parsed
        as a valid HTTP/1.0 or 1.1 status line, passed into
        :class:`http.client.HTTPConnection`.

        .. note::
           Only works in Python 2. This parameter is ignored in Python 3.

    :param timeout:
        Socket timeout in seconds for each individual connection. This can
        be a float or integer, which sets the timeout for the HTTP request,
        or an instance of :class:`urllib3.util.Timeout` which gives you more
        fine-grained control over request timeouts. After the constructor has
        been parsed, this is always a `urllib3.util.Timeout` object.

    :param maxsize:
        Number of connections to save that can be reused. More than 1 is useful
        in multithreaded situations. If ``block`` is set to False, more
        connections will be created but they will not be saved once they've
        been used.

    :param block:
        If set to True, no more than ``maxsize`` connections will be used at
        a time. When no free connections are available, the call will block
        until a connection has been released. This is a useful side effect for
        particular multithreaded situations where one does not want to use more
        than maxsize connections per host to prevent flooding.

    :param headers:
        Headers to include with all requests, unless other headers are given
        explicitly.

    :param retries:
        Retry configuration to use by default with requests in this pool.

    :param _proxy:
        Parsed proxy URL, should not be used directly, instead, see
        :class:`urllib3.ProxyManager`

    :param _proxy_headers:
        A dictionary with proxy headers, should not be used directly,
        instead, see :class:`urllib3.ProxyManager`

    :param \\**conn_kw:
        Additional parameters are used to create fresh :class:`urllib3.connection.HTTPConnection`,
        :class:`urllib3.connection.HTTPSConnection` instances.
    """
    scheme = 'http'
    ConnectionCls = HTTPConnection
    ResponseCls = HTTPResponse

    def __init__(self, host, port=None, strict=False, timeout=Timeout.DEFAULT_TIMEOUT, maxsize=1, block=False, headers=None, retries=None, _proxy=None, _proxy_headers=None, _proxy_config=None, **conn_kw):
        ConnectionPool.__init__(self, host, port)
        RequestMethods.__init__(self, headers)
        self.strict = strict
        if not isinstance(timeout, Timeout):
            timeout = Timeout.from_float(timeout)
        if retries is None:
            retries = Retry.DEFAULT
        self.timeout = timeout
        self.retries = retries
        self.pool = self.QueueCls(maxsize)
        self.block = block
        self.proxy = _proxy
        self.proxy_headers = _proxy_headers or {}
        self.proxy_config = _proxy_config
        for _ in xrange(maxsize):
            self.pool.put(None)
        self.num_connections = 0
        self.num_requests = 0
        self.conn_kw = conn_kw
        if self.proxy:
            self.conn_kw.setdefault('socket_options', [])
            self.conn_kw['proxy'] = self.proxy
            self.conn_kw['proxy_config'] = self.proxy_config

    def _new_conn(self):
        """
        Return a fresh :class:`HTTPConnection`.
        """
        self.num_connections += 1
        log.debug('Starting new HTTP connection (%d): %s:%s', self.num_connections, self.host, self.port or '80')
        conn = self.ConnectionCls(host=self.host, port=self.port, timeout=self.timeout.connect_timeout, strict=self.strict, **self.conn_kw)
        return conn

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

    def _put_conn(self, conn):
        """
        Put a connection back into the pool.

        :param conn:
            Connection object for the current host and port as returned by
            :meth:`._new_conn` or :meth:`._get_conn`.

        If the pool is already full, the connection is closed and discarded
        because we exceeded maxsize. If connections are discarded frequently,
        then maxsize should be increased.

        If the pool is closed, then the connection will be closed and discarded.
        """
        try:
            self.pool.put(conn, block=False)
            return
        except AttributeError:
            pass
        except queue.Full:
            log.warning('Connection pool is full, discarding connection: %s. Connection pool size: %s', self.host, self.pool.qsize())
        if conn:
            conn.close()

    def _validate_conn(self, conn):
        """
        Called right before a request is made, after the socket is created.
        """
        pass

    def _prepare_proxy(self, conn):
        pass

    def _get_timeout(self, timeout):
        """Helper that always returns a :class:`urllib3.util.Timeout`"""
        if timeout is _Default:
            return self.timeout.clone()
        if isinstance(timeout, Timeout):
            return timeout.clone()
        else:
            return Timeout.from_float(timeout)

    def _raise_timeout(self, err, url, timeout_value):
        """Is the error actually a timeout? Will raise a ReadTimeout or pass"""
        if isinstance(err, SocketTimeout):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if hasattr(err, 'errno') and err.errno in _blocking_errnos:
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
        if 'timed out' in str(err) or 'did not complete (read)' in str(err):
            raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)

    def _make_request(self, conn, method, url, timeout=_Default, chunked=False, **httplib_request_kw):
        """
        Perform a request on a given urllib connection object taken from our
        pool.

        :param conn:
            a connection from one of our connection pools

        :param timeout:
            Socket timeout in seconds for the request. This can be a
            float or integer, which will set the same timeout value for
            the socket connect and the socket read, or an instance of
            :class:`urllib3.util.Timeout`, which gives you more fine-grained
            control over your timeouts.
        """
        self.num_requests += 1
        timeout_obj = self._get_timeout(timeout)
        timeout_obj.start_connect()
        conn.timeout = timeout_obj.connect_timeout
        try:
            self._validate_conn(conn)
        except (SocketTimeout, BaseSSLError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=conn.timeout)
            raise
        try:
            if chunked:
                conn.request_chunked(method, url, **httplib_request_kw)
            else:
                conn.request(method, url, **httplib_request_kw)
        except BrokenPipeError:
            pass
        except IOError as e:
            if e.errno not in {errno.EPIPE, errno.ESHUTDOWN, errno.EPROTOTYPE}:
                raise
        read_timeout = timeout_obj.read_timeout
        if getattr(conn, 'sock', None):
            if read_timeout == 0:
                raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % read_timeout)
            if read_timeout is Timeout.DEFAULT_TIMEOUT:
                conn.sock.settimeout(socket.getdefaulttimeout())
            else:
                conn.sock.settimeout(read_timeout)
        try:
            try:
                httplib_response = conn.getresponse(buffering=True)
            except TypeError:
                try:
                    httplib_response = conn.getresponse()
                except BaseException as e:
                    six.raise_from(e, None)
        except (SocketTimeout, BaseSSLError, SocketError) as e:
            self._raise_timeout(err=e, url=url, timeout_value=read_timeout)
            raise
        http_version = getattr(conn, '_http_vsn_str', 'HTTP/?')
        log.debug('%s://%s:%s "%s %s %s" %s %s', self.scheme, self.host, self.port, method, url, http_version, httplib_response.status, httplib_response.length)
        try:
            assert_header_parsing(httplib_response.msg)
        except (HeaderParsingError, TypeError) as hpe:
            log.warning('Failed to parse headers (url=%s): %s', self._absolute_url(url), hpe, exc_info=True)
        return httplib_response

    def _absolute_url(self, path):
        return Url(scheme=self.scheme, host=self.host, port=self.port, path=path).url

    def close(self):
        """
        Close all pooled connections and disable the pool.
        """
        if self.pool is None:
            return
        old_pool, self.pool = (self.pool, None)
        try:
            while True:
                conn = old_pool.get(block=False)
                if conn:
                    conn.close()
        except queue.Empty:
            pass

    def is_same_host(self, url):
        """
        Check if the given ``url`` is a member of the same host as this
        connection pool.
        """
        if url.startswith('/'):
            return True
        scheme, host, port = get_host(url)
        if host is not None:
            host = _normalize_host(host, scheme=scheme)
        if self.port and (not port):
            port = port_by_scheme.get(scheme)
        elif not self.port and port == port_by_scheme.get(scheme):
            port = None
        return (scheme, host, port) == (self.scheme, self.host, self.port)

    def urlopen(self, method, url, body=None, headers=None, retries=None, redirect=True, assert_same_host=True, timeout=_Default, pool_timeout=None, release_conn=None, chunked=False, body_pos=None, **response_kw):
        """
        Get a connection from the pool and perform an HTTP request. This is the
        lowest level call for making a request, so you'll need to specify all
        the raw details.

        .. note::

           More commonly, it's appropriate to use a convenience method provided
           by :class:`.RequestMethods`, such as :meth:`request`.

        .. note::

           `release_conn` will only behave as expected if
           `preload_content=False` because we want to make
           `preload_content=False` the default behaviour someday soon without
           breaking backwards compatibility.

        :param method:
            HTTP request method (such as GET, POST, PUT, etc.)

        :param url:
            The URL to perform the request on.

        :param body:
            Data to send in the request body, either :class:`str`, :class:`bytes`,
            an iterable of :class:`str`/:class:`bytes`, or a file-like object.

        :param headers:
            Dictionary of custom headers to send, such as User-Agent,
            If-None-Match, etc. If None, pool headers are used. If provided,
            these headers completely replace any pool-specific headers.

        :param retries:
            Configure the number of retries to allow before raising a
            :class:`~urllib3.exceptions.MaxRetryError` exception.

            Pass ``None`` to retry until you receive a response. Pass a
            :class:`~urllib3.util.retry.Retry` object for fine-grained control
            over different types of retries.
            Pass an integer number to retry connection errors that many times,
            but no other types of errors. Pass zero to never retry.

            If ``False``, then retries are disabled and any exception is raised
            immediately. Also, instead of raising a MaxRetryError on redirects,
            the redirect response will be returned.

        :type retries: :class:`~urllib3.util.retry.Retry`, False, or an int.

        :param redirect:
            If True, automatically handle redirects (status codes 301, 302,
            303, 307, 308). Each redirect counts as a retry. Disabling retries
            will disable redirect, too.

        :param assert_same_host:
            If ``True``, will make sure that the host of the pool requests is
            consistent else will raise HostChangedError. When ``False``, you can
            use the pool on an HTTP proxy and request foreign hosts.

        :param timeout:
            If specified, overrides the default timeout for this one
            request. It may be a float (in seconds) or an instance of
            :class:`urllib3.util.Timeout`.

        :param pool_timeout:
            If set and the pool is set to block=True, then this method will
            block for ``pool_timeout`` seconds and raise EmptyPoolError if no
            connection is available within the time period.

        :param release_conn:
            If False, then the urlopen call will not release the connection
            back into the pool once a response is received (but will release if
            you read the entire contents of the response such as when
            `preload_content=True`). This is useful if you're not preloading
            the response's content immediately. You will need to call
            ``r.release_conn()`` on the response ``r`` to return the connection
            back into the pool. If None, it takes the value of
            ``response_kw.get('preload_content', True)``.

        :param chunked:
            If True, urllib3 will send the body using chunked transfer
            encoding. Otherwise, urllib3 will send the body using the standard
            content-length form. Defaults to False.

        :param int body_pos:
            Position to seek to in file-like body in the event of a retry or
            redirect. Typically this won't need to be set because urllib3 will
            auto-populate the value when needed.

        :param \\**response_kw:
            Additional parameters are passed to
            :meth:`urllib3.response.HTTPResponse.from_httplib`
        """
        parsed_url = parse_url(url)
        destination_scheme = parsed_url.scheme
        if headers is None:
            headers = self.headers
        if not isinstance(retries, Retry):
            retries = Retry.from_int(retries, redirect=redirect, default=self.retries)
        if release_conn is None:
            release_conn = response_kw.get('preload_content', True)
        if assert_same_host and (not self.is_same_host(url)):
            raise HostChangedError(self, url, retries)
        if url.startswith('/'):
            url = six.ensure_str(_encode_target(url))
        else:
            url = six.ensure_str(parsed_url.url)
        conn = None
        release_this_conn = release_conn
        http_tunnel_required = connection_requires_http_tunnel(self.proxy, self.proxy_config, destination_scheme)
        if not http_tunnel_required:
            headers = headers.copy()
            headers.update(self.proxy_headers)
        err = None
        clean_exit = False
        body_pos = set_file_position(body, body_pos)
        try:
            timeout_obj = self._get_timeout(timeout)
            conn = self._get_conn(timeout=pool_timeout)
            conn.timeout = timeout_obj.connect_timeout
            is_new_proxy_conn = self.proxy is not None and (not getattr(conn, 'sock', None))
            if is_new_proxy_conn and http_tunnel_required:
                self._prepare_proxy(conn)
            httplib_response = self._make_request(conn, method, url, timeout=timeout_obj, body=body, headers=headers, chunked=chunked)
            response_conn = conn if not release_conn else None
            response_kw['request_method'] = method
            response = self.ResponseCls.from_httplib(httplib_response, pool=self, connection=response_conn, retries=retries, **response_kw)
            clean_exit = True
        except EmptyPoolError:
            clean_exit = True
            release_this_conn = False
            raise
        except (TimeoutError, HTTPException, SocketError, ProtocolError, BaseSSLError, SSLError, CertificateError) as e:
            clean_exit = False

            def _is_ssl_error_message_from_http_proxy(ssl_error):
                message = ' '.join(re.split('[^a-z]', str(ssl_error).lower()))
                return 'wrong version number' in message or 'unknown protocol' in message
            if isinstance(e, BaseSSLError) and self.proxy and _is_ssl_error_message_from_http_proxy(e):
                e = ProxyError('Your proxy appears to only use HTTP and not HTTPS, try changing your proxy URL to be HTTP. See: https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#https-proxy-error-http-proxy', SSLError(e))
            elif isinstance(e, (BaseSSLError, CertificateError)):
                e = SSLError(e)
            elif isinstance(e, (SocketError, NewConnectionError)) and self.proxy:
                e = ProxyError('Cannot connect to proxy.', e)
            elif isinstance(e, (SocketError, HTTPException)):
                e = ProtocolError('Connection aborted.', e)
            retries = retries.increment(method, url, error=e, _pool=self, _stacktrace=sys.exc_info()[2])
            retries.sleep()
            err = e
        finally:
            if not clean_exit:
                conn = conn and conn.close()
                release_this_conn = True
            if release_this_conn:
                self._put_conn(conn)
        if not conn:
            log.warning("Retrying (%r) after connection broken by '%r': %s", retries, err, url)
            return self.urlopen(method, url, body, headers, retries, redirect, assert_same_host, timeout=timeout, pool_timeout=pool_timeout, release_conn=release_conn, chunked=chunked, body_pos=body_pos, **response_kw)
        redirect_location = redirect and response.get_redirect_location()
        if redirect_location:
            if response.status == 303:
                method = 'GET'
            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_redirect:
                    response.drain_conn()
                    raise
                return response
            response.drain_conn()
            retries.sleep_for_retry(response)
            log.debug('Redirecting %s -> %s', url, redirect_location)
            return self.urlopen(method, redirect_location, body, headers, retries=retries, redirect=redirect, assert_same_host=assert_same_host, timeout=timeout, pool_timeout=pool_timeout, release_conn=release_conn, chunked=chunked, body_pos=body_pos, **response_kw)
        has_retry_after = bool(response.getheader('Retry-After'))
        if retries.is_retry(method, response.status, has_retry_after):
            try:
                retries = retries.increment(method, url, response=response, _pool=self)
            except MaxRetryError:
                if retries.raise_on_status:
                    response.drain_conn()
                    raise
                return response
            response.drain_conn()
            retries.sleep(response)
            log.debug('Retry: %s', url)
            return self.urlopen(method, url, body, headers, retries=retries, redirect=redirect, assert_same_host=assert_same_host, timeout=timeout, pool_timeout=pool_timeout, release_conn=release_conn, chunked=chunked, body_pos=body_pos, **response_kw)
        return response