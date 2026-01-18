import asyncio
import functools
import random
import sys
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import suppress
from http import HTTPStatus
from http.cookies import SimpleCookie
from itertools import cycle, islice
from time import monotonic
from types import TracebackType
from typing import (
import attr
from . import hdrs, helpers
from .abc import AbstractResolver
from .client_exceptions import (
from .client_proto import ResponseHandler
from .client_reqrep import ClientRequest, Fingerprint, _merge_ssl_params
from .helpers import ceil_timeout, get_running_loop, is_ip_address, noop, sentinel
from .locks import EventResultOrError
from .resolver import DefaultResolver
class TCPConnector(BaseConnector):
    """TCP connector.

    verify_ssl - Set to True to check ssl certifications.
    fingerprint - Pass the binary sha256
        digest of the expected certificate in DER format to verify
        that the certificate the server presents matches. See also
        https://en.wikipedia.org/wiki/Transport_Layer_Security#Certificate_pinning
    resolver - Enable DNS lookups and use this
        resolver
    use_dns_cache - Use memory cache for DNS lookups.
    ttl_dns_cache - Max seconds having cached a DNS entry, None forever.
    family - socket address family
    local_addr - local tuple of (host, port) to bind socket to

    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    enable_cleanup_closed - Enables clean-up closed ssl transports.
                            Disabled by default.
    loop - Optional event loop.
    """

    def __init__(self, *, verify_ssl: bool=True, fingerprint: Optional[bytes]=None, use_dns_cache: bool=True, ttl_dns_cache: Optional[int]=10, family: int=0, ssl_context: Optional[SSLContext]=None, ssl: Union[bool, Fingerprint, SSLContext]=True, local_addr: Optional[Tuple[str, int]]=None, resolver: Optional[AbstractResolver]=None, keepalive_timeout: Union[None, float, object]=sentinel, force_close: bool=False, limit: int=100, limit_per_host: int=0, enable_cleanup_closed: bool=False, loop: Optional[asyncio.AbstractEventLoop]=None, timeout_ceil_threshold: float=5):
        super().__init__(keepalive_timeout=keepalive_timeout, force_close=force_close, limit=limit, limit_per_host=limit_per_host, enable_cleanup_closed=enable_cleanup_closed, loop=loop, timeout_ceil_threshold=timeout_ceil_threshold)
        self._ssl = _merge_ssl_params(ssl, verify_ssl, ssl_context, fingerprint)
        if resolver is None:
            resolver = DefaultResolver(loop=self._loop)
        self._resolver = resolver
        self._use_dns_cache = use_dns_cache
        self._cached_hosts = _DNSCacheTable(ttl=ttl_dns_cache)
        self._throttle_dns_events: Dict[Tuple[str, int], EventResultOrError] = {}
        self._family = family
        self._local_addr = local_addr

    def close(self) -> Awaitable[None]:
        """Close all ongoing DNS calls."""
        for ev in self._throttle_dns_events.values():
            ev.cancel()
        return super().close()

    @property
    def family(self) -> int:
        """Socket family like AF_INET."""
        return self._family

    @property
    def use_dns_cache(self) -> bool:
        """True if local DNS caching is enabled."""
        return self._use_dns_cache

    def clear_dns_cache(self, host: Optional[str]=None, port: Optional[int]=None) -> None:
        """Remove specified host/port or clear all dns local cache."""
        if host is not None and port is not None:
            self._cached_hosts.remove((host, port))
        elif host is not None or port is not None:
            raise ValueError('either both host and port or none of them are allowed')
        else:
            self._cached_hosts.clear()

    async def _resolve_host(self, host: str, port: int, traces: Optional[List['Trace']]=None) -> List[Dict[str, Any]]:
        if is_ip_address(host):
            return [{'hostname': host, 'host': host, 'port': port, 'family': self._family, 'proto': 0, 'flags': 0}]
        if not self._use_dns_cache:
            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_start(host)
            res = await self._resolver.resolve(host, port, family=self._family)
            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_end(host)
            return res
        key = (host, port)
        if key in self._cached_hosts and (not self._cached_hosts.expired(key)):
            result = self._cached_hosts.next_addrs(key)
            if traces:
                for trace in traces:
                    await trace.send_dns_cache_hit(host)
            return result
        if key in self._throttle_dns_events:
            event = self._throttle_dns_events[key]
            if traces:
                for trace in traces:
                    await trace.send_dns_cache_hit(host)
            await event.wait()
        else:
            self._throttle_dns_events[key] = EventResultOrError(self._loop)
            if traces:
                for trace in traces:
                    await trace.send_dns_cache_miss(host)
            try:
                if traces:
                    for trace in traces:
                        await trace.send_dns_resolvehost_start(host)
                addrs = await self._resolver.resolve(host, port, family=self._family)
                if traces:
                    for trace in traces:
                        await trace.send_dns_resolvehost_end(host)
                self._cached_hosts.add(key, addrs)
                self._throttle_dns_events[key].set()
            except BaseException as e:
                self._throttle_dns_events[key].set(exc=e)
                raise
            finally:
                self._throttle_dns_events.pop(key)
        return self._cached_hosts.next_addrs(key)

    async def _create_connection(self, req: ClientRequest, traces: List['Trace'], timeout: 'ClientTimeout') -> ResponseHandler:
        """Create connection.

        Has same keyword arguments as BaseEventLoop.create_connection.
        """
        if req.proxy:
            _, proto = await self._create_proxy_connection(req, traces, timeout)
        else:
            _, proto = await self._create_direct_connection(req, traces, timeout)
        return proto

    @staticmethod
    @functools.lru_cache(None)
    def _make_ssl_context(verified: bool) -> SSLContext:
        if verified:
            return ssl.create_default_context()
        else:
            sslcontext = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            sslcontext.options |= ssl.OP_NO_SSLv2
            sslcontext.options |= ssl.OP_NO_SSLv3
            sslcontext.check_hostname = False
            sslcontext.verify_mode = ssl.CERT_NONE
            try:
                sslcontext.options |= ssl.OP_NO_COMPRESSION
            except AttributeError as attr_err:
                warnings.warn('{!s}: The Python interpreter is compiled against OpenSSL < 1.0.0. Ref: https://docs.python.org/3/library/ssl.html#ssl.OP_NO_COMPRESSION'.format(attr_err))
            sslcontext.set_default_verify_paths()
            return sslcontext

    def _get_ssl_context(self, req: ClientRequest) -> Optional[SSLContext]:
        """Logic to get the correct SSL context

        0. if req.ssl is false, return None

        1. if ssl_context is specified in req, use it
        2. if _ssl_context is specified in self, use it
        3. otherwise:
            1. if verify_ssl is not specified in req, use self.ssl_context
               (will generate a default context according to self.verify_ssl)
            2. if verify_ssl is True in req, generate a default SSL context
            3. if verify_ssl is False in req, generate a SSL context that
               won't verify
        """
        if req.is_ssl():
            if ssl is None:
                raise RuntimeError('SSL is not supported.')
            sslcontext = req.ssl
            if isinstance(sslcontext, ssl.SSLContext):
                return sslcontext
            if sslcontext is not True:
                return self._make_ssl_context(False)
            sslcontext = self._ssl
            if isinstance(sslcontext, ssl.SSLContext):
                return sslcontext
            if sslcontext is not True:
                return self._make_ssl_context(False)
            return self._make_ssl_context(True)
        else:
            return None

    def _get_fingerprint(self, req: ClientRequest) -> Optional['Fingerprint']:
        ret = req.ssl
        if isinstance(ret, Fingerprint):
            return ret
        ret = self._ssl
        if isinstance(ret, Fingerprint):
            return ret
        return None

    async def _wrap_create_connection(self, *args: Any, req: ClientRequest, timeout: 'ClientTimeout', client_error: Type[Exception]=ClientConnectorError, **kwargs: Any) -> Tuple[asyncio.Transport, ResponseHandler]:
        try:
            async with ceil_timeout(timeout.sock_connect, ceil_threshold=timeout.ceil_threshold):
                return await self._loop.create_connection(*args, **kwargs)
        except cert_errors as exc:
            raise ClientConnectorCertificateError(req.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(req.connection_key, exc) from exc
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise client_error(req.connection_key, exc) from exc

    def _fail_on_no_start_tls(self, req: 'ClientRequest') -> None:
        """Raise a :py:exc:`RuntimeError` on missing ``start_tls()``.

        It is necessary for TLS-in-TLS so that it is possible to
        send HTTPS queries through HTTPS proxies.

        This doesn't affect regular HTTP requests, though.
        """
        if not req.is_ssl():
            return
        proxy_url = req.proxy
        assert proxy_url is not None
        if proxy_url.scheme != 'https':
            return
        self._check_loop_for_start_tls()

    def _check_loop_for_start_tls(self) -> None:
        try:
            self._loop.start_tls
        except AttributeError as attr_exc:
            raise RuntimeError('An HTTPS request is being sent through an HTTPS proxy. This needs support for TLS in TLS but it is not implemented in your runtime for the stdlib asyncio.\n\nPlease upgrade to Python 3.11 or higher. For more details, please see:\n* https://bugs.python.org/issue37179\n* https://github.com/python/cpython/pull/28073\n* https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support\n* https://github.com/aio-libs/aiohttp/discussions/6044\n') from attr_exc

    def _loop_supports_start_tls(self) -> bool:
        try:
            self._check_loop_for_start_tls()
        except RuntimeError:
            return False
        else:
            return True

    def _warn_about_tls_in_tls(self, underlying_transport: asyncio.Transport, req: ClientRequest) -> None:
        """Issue a warning if the requested URL has HTTPS scheme."""
        if req.request_info.url.scheme != 'https':
            return
        asyncio_supports_tls_in_tls = getattr(underlying_transport, '_start_tls_compatible', False)
        if asyncio_supports_tls_in_tls:
            return
        warnings.warn("An HTTPS request is being sent through an HTTPS proxy. This support for TLS in TLS is known to be disabled in the stdlib asyncio (Python <3.11). This is why you'll probably see an error in the log below.\n\nIt is possible to enable it via monkeypatching. For more details, see:\n* https://bugs.python.org/issue37179\n* https://github.com/python/cpython/pull/28073\n\nYou can temporarily patch this as follows:\n* https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support\n* https://github.com/aio-libs/aiohttp/discussions/6044\n", RuntimeWarning, source=self, stacklevel=3)

    async def _start_tls_connection(self, underlying_transport: asyncio.Transport, req: ClientRequest, timeout: 'ClientTimeout', client_error: Type[Exception]=ClientConnectorError) -> Tuple[asyncio.BaseTransport, ResponseHandler]:
        """Wrap the raw TCP transport with TLS."""
        tls_proto = self._factory()
        sslcontext = cast(ssl.SSLContext, self._get_ssl_context(req))
        try:
            async with ceil_timeout(timeout.sock_connect, ceil_threshold=timeout.ceil_threshold):
                try:
                    tls_transport = await self._loop.start_tls(underlying_transport, tls_proto, sslcontext, server_hostname=req.server_hostname or req.host, ssl_handshake_timeout=timeout.total)
                except BaseException:
                    underlying_transport.close()
                    raise
        except cert_errors as exc:
            raise ClientConnectorCertificateError(req.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(req.connection_key, exc) from exc
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise client_error(req.connection_key, exc) from exc
        except TypeError as type_err:
            raise ClientConnectionError(f'Cannot initialize a TLS-in-TLS connection to host {req.host!s}:{req.port:d} through an underlying connection to an HTTPS proxy {req.proxy!s} ssl:{req.ssl or 'default'} [{type_err!s}]') from type_err
        else:
            if tls_transport is None:
                msg = 'Failed to start TLS (possibly caused by closing transport)'
                raise client_error(req.connection_key, OSError(msg))
            tls_proto.connection_made(tls_transport)
        return (tls_transport, tls_proto)

    async def _create_direct_connection(self, req: ClientRequest, traces: List['Trace'], timeout: 'ClientTimeout', *, client_error: Type[Exception]=ClientConnectorError) -> Tuple[asyncio.Transport, ResponseHandler]:
        sslcontext = self._get_ssl_context(req)
        fingerprint = self._get_fingerprint(req)
        host = req.url.raw_host
        assert host is not None
        if host.endswith('..'):
            host = host.rstrip('.') + '.'
        port = req.port
        assert port is not None
        host_resolved = asyncio.ensure_future(self._resolve_host(host, port, traces=traces), loop=self._loop)
        try:
            hosts = await asyncio.shield(host_resolved)
        except asyncio.CancelledError:

            def drop_exception(fut: 'asyncio.Future[List[Dict[str, Any]]]') -> None:
                with suppress(Exception, asyncio.CancelledError):
                    fut.result()
            host_resolved.add_done_callback(drop_exception)
            raise
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise ClientConnectorError(req.connection_key, exc) from exc
        last_exc: Optional[Exception] = None
        for hinfo in hosts:
            host = hinfo['host']
            port = hinfo['port']
            server_hostname = (req.server_hostname or hinfo['hostname']).rstrip('.') if sslcontext else None
            try:
                transp, proto = await self._wrap_create_connection(self._factory, host, port, timeout=timeout, ssl=sslcontext, family=hinfo['family'], proto=hinfo['proto'], flags=hinfo['flags'], server_hostname=server_hostname, local_addr=self._local_addr, req=req, client_error=client_error)
            except ClientConnectorError as exc:
                last_exc = exc
                continue
            if req.is_ssl() and fingerprint:
                try:
                    fingerprint.check(transp)
                except ServerFingerprintMismatch as exc:
                    transp.close()
                    if not self._cleanup_closed_disabled:
                        self._cleanup_closed_transports.append(transp)
                    last_exc = exc
                    continue
            return (transp, proto)
        else:
            assert last_exc is not None
            raise last_exc

    async def _create_proxy_connection(self, req: ClientRequest, traces: List['Trace'], timeout: 'ClientTimeout') -> Tuple[asyncio.BaseTransport, ResponseHandler]:
        self._fail_on_no_start_tls(req)
        runtime_has_start_tls = self._loop_supports_start_tls()
        headers: Dict[str, str] = {}
        if req.proxy_headers is not None:
            headers = req.proxy_headers
        headers[hdrs.HOST] = req.headers[hdrs.HOST]
        url = req.proxy
        assert url is not None
        proxy_req = ClientRequest(hdrs.METH_GET, url, headers=headers, auth=req.proxy_auth, loop=self._loop, ssl=req.ssl)
        transport, proto = await self._create_direct_connection(proxy_req, [], timeout, client_error=ClientProxyConnectionError)
        proto.force_close()
        auth = proxy_req.headers.pop(hdrs.AUTHORIZATION, None)
        if auth is not None:
            if not req.is_ssl():
                req.headers[hdrs.PROXY_AUTHORIZATION] = auth
            else:
                proxy_req.headers[hdrs.PROXY_AUTHORIZATION] = auth
        if req.is_ssl():
            if runtime_has_start_tls:
                self._warn_about_tls_in_tls(transport, req)
            proxy_req.method = hdrs.METH_CONNECT
            proxy_req.url = req.url
            key = attr.evolve(req.connection_key, proxy=None, proxy_auth=None, proxy_headers_hash=None)
            conn = Connection(self, key, proto, self._loop)
            proxy_resp = await proxy_req.send(conn)
            try:
                protocol = conn._protocol
                assert protocol is not None
                protocol.set_response_params(read_until_eof=runtime_has_start_tls, timeout_ceil_threshold=self._timeout_ceil_threshold)
                resp = await proxy_resp.start(conn)
            except BaseException:
                proxy_resp.close()
                conn.close()
                raise
            else:
                conn._protocol = None
                conn._transport = None
                try:
                    if resp.status != 200:
                        message = resp.reason
                        if message is None:
                            message = HTTPStatus(resp.status).phrase
                        raise ClientHttpProxyError(proxy_resp.request_info, resp.history, status=resp.status, message=message, headers=resp.headers)
                    if not runtime_has_start_tls:
                        rawsock = transport.get_extra_info('socket', default=None)
                        if rawsock is None:
                            raise RuntimeError('Transport does not expose socket instance')
                        rawsock = rawsock.dup()
                except BaseException:
                    transport.close()
                    raise
                finally:
                    if not runtime_has_start_tls:
                        transport.close()
                if not runtime_has_start_tls:
                    sslcontext = self._get_ssl_context(req)
                    return await self._wrap_create_connection(self._factory, timeout=timeout, ssl=sslcontext, sock=rawsock, server_hostname=req.host, req=req)
                return await self._start_tls_connection(transport, req=req, timeout=timeout)
            finally:
                proxy_resp.close()
        return (transport, proto)