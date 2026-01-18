from __future__ import annotations
import contextlib
import typing
from types import TracebackType
import httpcore
from .._config import DEFAULT_LIMITS, Limits, Proxy, create_ssl_context
from .._exceptions import (
from .._models import Request, Response
from .._types import AsyncByteStream, CertTypes, ProxyTypes, SyncByteStream, VerifyTypes
from .._urls import URL
from .base import AsyncBaseTransport, BaseTransport
class AsyncHTTPTransport(AsyncBaseTransport):

    def __init__(self, verify: VerifyTypes=True, cert: CertTypes | None=None, http1: bool=True, http2: bool=False, limits: Limits=DEFAULT_LIMITS, trust_env: bool=True, proxy: ProxyTypes | None=None, uds: str | None=None, local_address: str | None=None, retries: int=0, socket_options: typing.Iterable[SOCKET_OPTION] | None=None) -> None:
        ssl_context = create_ssl_context(verify=verify, cert=cert, trust_env=trust_env)
        proxy = Proxy(url=proxy) if isinstance(proxy, (str, URL)) else proxy
        if proxy is None:
            self._pool = httpcore.AsyncConnectionPool(ssl_context=ssl_context, max_connections=limits.max_connections, max_keepalive_connections=limits.max_keepalive_connections, keepalive_expiry=limits.keepalive_expiry, http1=http1, http2=http2, uds=uds, local_address=local_address, retries=retries, socket_options=socket_options)
        elif proxy.url.scheme in ('http', 'https'):
            self._pool = httpcore.AsyncHTTPProxy(proxy_url=httpcore.URL(scheme=proxy.url.raw_scheme, host=proxy.url.raw_host, port=proxy.url.port, target=proxy.url.raw_path), proxy_auth=proxy.raw_auth, proxy_headers=proxy.headers.raw, ssl_context=ssl_context, max_connections=limits.max_connections, max_keepalive_connections=limits.max_keepalive_connections, keepalive_expiry=limits.keepalive_expiry, http1=http1, http2=http2, socket_options=socket_options)
        elif proxy.url.scheme == 'socks5':
            try:
                import socksio
            except ImportError:
                raise ImportError("Using SOCKS proxy, but the 'socksio' package is not installed. Make sure to install httpx using `pip install httpx[socks]`.") from None
            self._pool = httpcore.AsyncSOCKSProxy(proxy_url=httpcore.URL(scheme=proxy.url.raw_scheme, host=proxy.url.raw_host, port=proxy.url.port, target=proxy.url.raw_path), proxy_auth=proxy.raw_auth, ssl_context=ssl_context, max_connections=limits.max_connections, max_keepalive_connections=limits.max_keepalive_connections, keepalive_expiry=limits.keepalive_expiry, http1=http1, http2=http2)
        else:
            raise ValueError("Proxy protocol must be either 'http', 'https', or 'socks5', but got {proxy.url.scheme!r}.")

    async def __aenter__(self: A) -> A:
        await self._pool.__aenter__()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None=None, exc_value: BaseException | None=None, traceback: TracebackType | None=None) -> None:
        with map_httpcore_exceptions():
            await self._pool.__aexit__(exc_type, exc_value, traceback)

    async def handle_async_request(self, request: Request) -> Response:
        assert isinstance(request.stream, AsyncByteStream)
        req = httpcore.Request(method=request.method, url=httpcore.URL(scheme=request.url.raw_scheme, host=request.url.raw_host, port=request.url.port, target=request.url.raw_path), headers=request.headers.raw, content=request.stream, extensions=request.extensions)
        with map_httpcore_exceptions():
            resp = await self._pool.handle_async_request(req)
        assert isinstance(resp.stream, typing.AsyncIterable)
        return Response(status_code=resp.status, headers=resp.headers, stream=AsyncResponseStream(resp.stream), extensions=resp.extensions)

    async def aclose(self) -> None:
        await self._pool.aclose()