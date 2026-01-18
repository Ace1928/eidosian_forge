from __future__ import annotations
import asyncio
import email.utils
import functools
import http
import inspect
import logging
import socket
import warnings
from types import TracebackType
from typing import (
from ..datastructures import Headers, HeadersLike, MultipleValuesError
from ..exceptions import (
from ..extensions import Extension, ServerExtensionFactory
from ..extensions.permessage_deflate import enable_server_permessage_deflate
from ..headers import (
from ..http import USER_AGENT
from ..protocol import State
from ..typing import ExtensionHeader, LoggerLike, Origin, StatusLike, Subprotocol
from .compatibility import asyncio_timeout
from .handshake import build_response, check_request
from .http import read_request
from .protocol import WebSocketCommonProtocol
class WebSocketServerProtocol(WebSocketCommonProtocol):
    """
    WebSocket server connection.

    :class:`WebSocketServerProtocol` provides :meth:`recv` and :meth:`send`
    coroutines for receiving and sending messages.

    It supports asynchronous iteration to receive messages::

        async for message in websocket:
            await process(message)

    The iterator exits normally when the connection is closed with close code
    1000 (OK) or 1001 (going away) or without a close code. It raises
    a :exc:`~websockets.exceptions.ConnectionClosedError` when the connection
    is closed with any other code.

    You may customize the opening handshake in a subclass by
    overriding :meth:`process_request` or :meth:`select_subprotocol`.

    Args:
        ws_server: WebSocket server that created this connection.

    See :func:`serve` for the documentation of ``ws_handler``, ``logger``, ``origins``,
    ``extensions``, ``subprotocols``, ``extra_headers``, and ``server_header``.

    See :class:`~websockets.legacy.protocol.WebSocketCommonProtocol` for the
    documentation of ``ping_interval``, ``ping_timeout``, ``close_timeout``,
    ``max_size``, ``max_queue``, ``read_limit``, and ``write_limit``.

    """
    is_client = False
    side = 'server'

    def __init__(self, ws_handler: Union[Callable[[WebSocketServerProtocol], Awaitable[Any]], Callable[[WebSocketServerProtocol, str], Awaitable[Any]]], ws_server: WebSocketServer, *, logger: Optional[LoggerLike]=None, origins: Optional[Sequence[Optional[Origin]]]=None, extensions: Optional[Sequence[ServerExtensionFactory]]=None, subprotocols: Optional[Sequence[Subprotocol]]=None, extra_headers: Optional[HeadersLikeOrCallable]=None, server_header: Optional[str]=USER_AGENT, process_request: Optional[Callable[[str, Headers], Awaitable[Optional[HTTPResponse]]]]=None, select_subprotocol: Optional[Callable[[Sequence[Subprotocol], Sequence[Subprotocol]], Subprotocol]]=None, open_timeout: Optional[float]=10, **kwargs: Any) -> None:
        if logger is None:
            logger = logging.getLogger('websockets.server')
        super().__init__(logger=logger, **kwargs)
        if origins is not None and '' in origins:
            warnings.warn("use None instead of '' in origins", DeprecationWarning)
            origins = [None if origin == '' else origin for origin in origins]
        self.ws_handler = remove_path_argument(ws_handler)
        self.ws_server = ws_server
        self.origins = origins
        self.available_extensions = extensions
        self.available_subprotocols = subprotocols
        self.extra_headers = extra_headers
        self.server_header = server_header
        self._process_request = process_request
        self._select_subprotocol = select_subprotocol
        self.open_timeout = open_timeout

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        """
        Register connection and initialize a task to handle it.

        """
        super().connection_made(transport)
        self.ws_server.register(self)
        self.handler_task = self.loop.create_task(self.handler())

    async def handler(self) -> None:
        """
        Handle the lifecycle of a WebSocket connection.

        Since this method doesn't have a caller able to handle exceptions, it
        attempts to log relevant ones and guarantees that the TCP connection is
        closed before exiting.

        """
        try:
            try:
                async with asyncio_timeout(self.open_timeout):
                    await self.handshake(origins=self.origins, available_extensions=self.available_extensions, available_subprotocols=self.available_subprotocols, extra_headers=self.extra_headers)
            except asyncio.TimeoutError:
                raise
            except ConnectionError:
                raise
            except Exception as exc:
                if isinstance(exc, AbortHandshake):
                    status, headers, body = (exc.status, exc.headers, exc.body)
                elif isinstance(exc, InvalidOrigin):
                    if self.debug:
                        self.logger.debug('! invalid origin', exc_info=True)
                    status, headers, body = (http.HTTPStatus.FORBIDDEN, Headers(), f'Failed to open a WebSocket connection: {exc}.\n'.encode())
                elif isinstance(exc, InvalidUpgrade):
                    if self.debug:
                        self.logger.debug('! invalid upgrade', exc_info=True)
                    status, headers, body = (http.HTTPStatus.UPGRADE_REQUIRED, Headers([('Upgrade', 'websocket')]), f'Failed to open a WebSocket connection: {exc}.\n\nYou cannot access a WebSocket server directly with a browser. You need a WebSocket client.\n'.encode())
                elif isinstance(exc, InvalidHandshake):
                    if self.debug:
                        self.logger.debug('! invalid handshake', exc_info=True)
                    status, headers, body = (http.HTTPStatus.BAD_REQUEST, Headers(), f'Failed to open a WebSocket connection: {exc}.\n'.encode())
                else:
                    self.logger.error('opening handshake failed', exc_info=True)
                    status, headers, body = (http.HTTPStatus.INTERNAL_SERVER_ERROR, Headers(), b'Failed to open a WebSocket connection.\nSee server log for more information.\n')
                headers.setdefault('Date', email.utils.formatdate(usegmt=True))
                if self.server_header is not None:
                    headers.setdefault('Server', self.server_header)
                headers.setdefault('Content-Length', str(len(body)))
                headers.setdefault('Content-Type', 'text/plain')
                headers.setdefault('Connection', 'close')
                self.write_http_response(status, headers, body)
                self.logger.info('connection rejected (%d %s)', status.value, status.phrase)
                await self.close_transport()
                return
            try:
                await self.ws_handler(self)
            except Exception:
                self.logger.error('connection handler failed', exc_info=True)
                if not self.closed:
                    self.fail_connection(1011)
                raise
            try:
                await self.close()
            except ConnectionError:
                raise
            except Exception:
                self.logger.error('closing handshake failed', exc_info=True)
                raise
        except Exception:
            try:
                self.transport.close()
            except Exception:
                pass
        finally:
            self.ws_server.unregister(self)
            self.logger.info('connection closed')

    async def read_http_request(self) -> Tuple[str, Headers]:
        """
        Read request line and headers from the HTTP request.

        If the request contains a body, it may be read from ``self.reader``
        after this coroutine returns.

        Raises:
            InvalidMessage: if the HTTP message is malformed or isn't an
                HTTP/1.1 GET request.

        """
        try:
            path, headers = await read_request(self.reader)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise InvalidMessage('did not receive a valid HTTP request') from exc
        if self.debug:
            self.logger.debug('< GET %s HTTP/1.1', path)
            for key, value in headers.raw_items():
                self.logger.debug('< %s: %s', key, value)
        self.path = path
        self.request_headers = headers
        return (path, headers)

    def write_http_response(self, status: http.HTTPStatus, headers: Headers, body: Optional[bytes]=None) -> None:
        """
        Write status line and headers to the HTTP response.

        This coroutine is also able to write a response body.

        """
        self.response_headers = headers
        if self.debug:
            self.logger.debug('> HTTP/1.1 %d %s', status.value, status.phrase)
            for key, value in headers.raw_items():
                self.logger.debug('> %s: %s', key, value)
            if body is not None:
                self.logger.debug('> [body] (%d bytes)', len(body))
        response = f'HTTP/1.1 {status.value} {status.phrase}\r\n'
        response += str(headers)
        self.transport.write(response.encode())
        if body is not None:
            self.transport.write(body)

    async def process_request(self, path: str, request_headers: Headers) -> Optional[HTTPResponse]:
        """
        Intercept the HTTP request and return an HTTP response if appropriate.

        You may override this method in a :class:`WebSocketServerProtocol`
        subclass, for example:

        * to return an HTTP 200 OK response on a given path; then a load
          balancer can use this path for a health check;
        * to authenticate the request and return an HTTP 401 Unauthorized or an
          HTTP 403 Forbidden when authentication fails.

        You may also override this method with the ``process_request``
        argument of :func:`serve` and :class:`WebSocketServerProtocol`. This
        is equivalent, except ``process_request`` won't have access to the
        protocol instance, so it can't store information for later use.

        :meth:`process_request` is expected to complete quickly. If it may run
        for a long time, then it should await :meth:`wait_closed` and exit if
        :meth:`wait_closed` completes, or else it could prevent the server
        from shutting down.

        Args:
            path: request path, including optional query string.
            request_headers: request headers.

        Returns:
            Optional[Tuple[StatusLike, HeadersLike, bytes]]: :obj:`None`
            to continue the WebSocket handshake normally.

            An HTTP response, represented by a 3-uple of the response status,
            headers, and body, to abort the WebSocket handshake and return
            that HTTP response instead.

        """
        if self._process_request is not None:
            response = self._process_request(path, request_headers)
            if isinstance(response, Awaitable):
                return await response
            else:
                warnings.warn('declare process_request as a coroutine', DeprecationWarning)
                return response
        return None

    @staticmethod
    def process_origin(headers: Headers, origins: Optional[Sequence[Optional[Origin]]]=None) -> Optional[Origin]:
        """
        Handle the Origin HTTP request header.

        Args:
            headers: request headers.
            origins: optional list of acceptable origins.

        Raises:
            InvalidOrigin: if the origin isn't acceptable.

        """
        try:
            origin = cast(Optional[Origin], headers.get('Origin'))
        except MultipleValuesError as exc:
            raise InvalidHeader('Origin', 'more than one Origin header found') from exc
        if origins is not None:
            if origin not in origins:
                raise InvalidOrigin(origin)
        return origin

    @staticmethod
    def process_extensions(headers: Headers, available_extensions: Optional[Sequence[ServerExtensionFactory]]) -> Tuple[Optional[str], List[Extension]]:
        """
        Handle the Sec-WebSocket-Extensions HTTP request header.

        Accept or reject each extension proposed in the client request.
        Negotiate parameters for accepted extensions.

        Return the Sec-WebSocket-Extensions HTTP response header and the list
        of accepted extensions.

        :rfc:`6455` leaves the rules up to the specification of each
        :extension.

        To provide this level of flexibility, for each extension proposed by
        the client, we check for a match with each extension available in the
        server configuration. If no match is found, the extension is ignored.

        If several variants of the same extension are proposed by the client,
        it may be accepted several times, which won't make sense in general.
        Extensions must implement their own requirements. For this purpose,
        the list of previously accepted extensions is provided.

        This process doesn't allow the server to reorder extensions. It can
        only select a subset of the extensions proposed by the client.

        Other requirements, for example related to mandatory extensions or the
        order of extensions, may be implemented by overriding this method.

        Args:
            headers: request headers.
            extensions: optional list of supported extensions.

        Raises:
            InvalidHandshake: to abort the handshake with an HTTP 400 error.

        """
        response_header_value: Optional[str] = None
        extension_headers: List[ExtensionHeader] = []
        accepted_extensions: List[Extension] = []
        header_values = headers.get_all('Sec-WebSocket-Extensions')
        if header_values and available_extensions:
            parsed_header_values: List[ExtensionHeader] = sum([parse_extension(header_value) for header_value in header_values], [])
            for name, request_params in parsed_header_values:
                for ext_factory in available_extensions:
                    if ext_factory.name != name:
                        continue
                    try:
                        response_params, extension = ext_factory.process_request_params(request_params, accepted_extensions)
                    except NegotiationError:
                        continue
                    extension_headers.append((name, response_params))
                    accepted_extensions.append(extension)
                    break
        if extension_headers:
            response_header_value = build_extension(extension_headers)
        return (response_header_value, accepted_extensions)

    def process_subprotocol(self, headers: Headers, available_subprotocols: Optional[Sequence[Subprotocol]]) -> Optional[Subprotocol]:
        """
        Handle the Sec-WebSocket-Protocol HTTP request header.

        Return Sec-WebSocket-Protocol HTTP response header, which is the same
        as the selected subprotocol.

        Args:
            headers: request headers.
            available_subprotocols: optional list of supported subprotocols.

        Raises:
            InvalidHandshake: to abort the handshake with an HTTP 400 error.

        """
        subprotocol: Optional[Subprotocol] = None
        header_values = headers.get_all('Sec-WebSocket-Protocol')
        if header_values and available_subprotocols:
            parsed_header_values: List[Subprotocol] = sum([parse_subprotocol(header_value) for header_value in header_values], [])
            subprotocol = self.select_subprotocol(parsed_header_values, available_subprotocols)
        return subprotocol

    def select_subprotocol(self, client_subprotocols: Sequence[Subprotocol], server_subprotocols: Sequence[Subprotocol]) -> Optional[Subprotocol]:
        """
        Pick a subprotocol among those supported by the client and the server.

        If several subprotocols are available, select the preferred subprotocol
        by giving equal weight to the preferences of the client and the server.

        If no subprotocol is available, proceed without a subprotocol.

        You may provide a ``select_subprotocol`` argument to :func:`serve` or
        :class:`WebSocketServerProtocol` to override this logic. For example,
        you could reject the handshake if the client doesn't support a
        particular subprotocol, rather than accept the handshake without that
        subprotocol.

        Args:
            client_subprotocols: list of subprotocols offered by the client.
            server_subprotocols: list of subprotocols available on the server.

        Returns:
            Optional[Subprotocol]: Selected subprotocol, if a common subprotocol
            was found.

            :obj:`None` to continue without a subprotocol.

        """
        if self._select_subprotocol is not None:
            return self._select_subprotocol(client_subprotocols, server_subprotocols)
        subprotocols = set(client_subprotocols) & set(server_subprotocols)
        if not subprotocols:
            return None
        return sorted(subprotocols, key=lambda p: client_subprotocols.index(p) + server_subprotocols.index(p))[0]

    async def handshake(self, origins: Optional[Sequence[Optional[Origin]]]=None, available_extensions: Optional[Sequence[ServerExtensionFactory]]=None, available_subprotocols: Optional[Sequence[Subprotocol]]=None, extra_headers: Optional[HeadersLikeOrCallable]=None) -> str:
        """
        Perform the server side of the opening handshake.

        Args:
            origins: list of acceptable values of the Origin HTTP header;
                include :obj:`None` if the lack of an origin is acceptable.
            extensions: list of supported extensions, in order in which they
                should be tried.
            subprotocols: list of supported subprotocols, in order of
                decreasing preference.
            extra_headers: arbitrary HTTP headers to add to the response when
                the handshake succeeds.

        Returns:
            str: path of the URI of the request.

        Raises:
            InvalidHandshake: if the handshake fails.

        """
        path, request_headers = await self.read_http_request()
        early_response_awaitable = self.process_request(path, request_headers)
        if isinstance(early_response_awaitable, Awaitable):
            early_response = await early_response_awaitable
        else:
            warnings.warn('declare process_request as a coroutine', DeprecationWarning)
            early_response = early_response_awaitable
        if self.state is State.CLOSED:
            raise BrokenPipeError('connection closed during opening handshake')
        if not self.ws_server.is_serving():
            early_response = (http.HTTPStatus.SERVICE_UNAVAILABLE, [], b'Server is shutting down.\n')
        if early_response is not None:
            raise AbortHandshake(*early_response)
        key = check_request(request_headers)
        self.origin = self.process_origin(request_headers, origins)
        extensions_header, self.extensions = self.process_extensions(request_headers, available_extensions)
        protocol_header = self.subprotocol = self.process_subprotocol(request_headers, available_subprotocols)
        response_headers = Headers()
        build_response(response_headers, key)
        if extensions_header is not None:
            response_headers['Sec-WebSocket-Extensions'] = extensions_header
        if protocol_header is not None:
            response_headers['Sec-WebSocket-Protocol'] = protocol_header
        if callable(extra_headers):
            extra_headers = extra_headers(path, self.request_headers)
        if extra_headers is not None:
            response_headers.update(extra_headers)
        response_headers.setdefault('Date', email.utils.formatdate(usegmt=True))
        if self.server_header is not None:
            response_headers.setdefault('Server', self.server_header)
        self.write_http_response(http.HTTPStatus.SWITCHING_PROTOCOLS, response_headers)
        self.logger.info('connection open')
        self.connection_open()
        return path