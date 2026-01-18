import abc
import asyncio
import base64
import hashlib
import os
import sys
import struct
import tornado
from urllib.parse import urlparse
import warnings
import zlib
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen, httpclient, httputil
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.iostream import StreamClosedError, IOStream
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado import simple_httpclient
from tornado.queues import Queue
from tornado.tcpclient import TCPClient
from tornado.util import _websocket_mask
from typing import (
from types import TracebackType
class WebSocketClientConnection(simple_httpclient._HTTPConnection):
    """WebSocket client connection.

    This class should not be instantiated directly; use the
    `websocket_connect` function instead.
    """
    protocol = None

    def __init__(self, request: httpclient.HTTPRequest, on_message_callback: Optional[Callable[[Union[None, str, bytes]], None]]=None, compression_options: Optional[Dict[str, Any]]=None, ping_interval: Optional[float]=None, ping_timeout: Optional[float]=None, max_message_size: int=_default_max_message_size, subprotocols: Optional[List[str]]=None, resolver: Optional[Resolver]=None) -> None:
        self.connect_future = Future()
        self.read_queue = Queue(1)
        self.key = base64.b64encode(os.urandom(16))
        self._on_message_callback = on_message_callback
        self.close_code = None
        self.close_reason = None
        self.params = _WebSocketParams(ping_interval=ping_interval, ping_timeout=ping_timeout, max_message_size=max_message_size, compression_options=compression_options)
        scheme, sep, rest = request.url.partition(':')
        scheme = {'ws': 'http', 'wss': 'https'}[scheme]
        request.url = scheme + sep + rest
        request.headers.update({'Upgrade': 'websocket', 'Connection': 'Upgrade', 'Sec-WebSocket-Key': self.key, 'Sec-WebSocket-Version': '13'})
        if subprotocols is not None:
            request.headers['Sec-WebSocket-Protocol'] = ','.join(subprotocols)
        if compression_options is not None:
            request.headers['Sec-WebSocket-Extensions'] = 'permessage-deflate; client_max_window_bits'
        request.follow_redirects = False
        self.tcp_client = TCPClient(resolver=resolver)
        super().__init__(None, request, lambda: None, self._on_http_response, 104857600, self.tcp_client, 65536, 104857600)

    def __del__(self) -> None:
        if self.protocol is not None:
            warnings.warn('Unclosed WebSocketClientConnection', ResourceWarning)

    def close(self, code: Optional[int]=None, reason: Optional[str]=None) -> None:
        """Closes the websocket connection.

        ``code`` and ``reason`` are documented under
        `WebSocketHandler.close`.

        .. versionadded:: 3.2

        .. versionchanged:: 4.0

           Added the ``code`` and ``reason`` arguments.
        """
        if self.protocol is not None:
            self.protocol.close(code, reason)
            self.protocol = None

    def on_connection_close(self) -> None:
        if not self.connect_future.done():
            self.connect_future.set_exception(StreamClosedError())
        self._on_message(None)
        self.tcp_client.close()
        super().on_connection_close()

    def on_ws_connection_close(self, close_code: Optional[int]=None, close_reason: Optional[str]=None) -> None:
        self.close_code = close_code
        self.close_reason = close_reason
        self.on_connection_close()

    def _on_http_response(self, response: httpclient.HTTPResponse) -> None:
        if not self.connect_future.done():
            if response.error:
                self.connect_future.set_exception(response.error)
            else:
                self.connect_future.set_exception(WebSocketError('Non-websocket response'))

    async def headers_received(self, start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], headers: httputil.HTTPHeaders) -> None:
        assert isinstance(start_line, httputil.ResponseStartLine)
        if start_line.code != 101:
            await super().headers_received(start_line, headers)
            return
        if self._timeout is not None:
            self.io_loop.remove_timeout(self._timeout)
            self._timeout = None
        self.headers = headers
        self.protocol = self.get_websocket_protocol()
        self.protocol._process_server_headers(self.key, self.headers)
        self.protocol.stream = self.connection.detach()
        IOLoop.current().add_callback(self.protocol._receive_frame_loop)
        self.protocol.start_pinging()
        self.final_callback = None
        future_set_result_unless_cancelled(self.connect_future, self)

    def write_message(self, message: Union[str, bytes, Dict[str, Any]], binary: bool=False) -> 'Future[None]':
        """Sends a message to the WebSocket server.

        If the stream is closed, raises `WebSocketClosedError`.
        Returns a `.Future` which can be used for flow control.

        .. versionchanged:: 5.0
           Exception raised on a closed stream changed from `.StreamClosedError`
           to `WebSocketClosedError`.
        """
        if self.protocol is None:
            raise WebSocketClosedError('Client connection has been closed')
        return self.protocol.write_message(message, binary=binary)

    def read_message(self, callback: Optional[Callable[['Future[Union[None, str, bytes]]'], None]]=None) -> Awaitable[Union[None, str, bytes]]:
        """Reads a message from the WebSocket server.

        If on_message_callback was specified at WebSocket
        initialization, this function will never return messages

        Returns a future whose result is the message, or None
        if the connection is closed.  If a callback argument
        is given it will be called with the future when it is
        ready.
        """
        awaitable = self.read_queue.get()
        if callback is not None:
            self.io_loop.add_future(asyncio.ensure_future(awaitable), callback)
        return awaitable

    def on_message(self, message: Union[str, bytes]) -> Optional[Awaitable[None]]:
        return self._on_message(message)

    def _on_message(self, message: Union[None, str, bytes]) -> Optional[Awaitable[None]]:
        if self._on_message_callback:
            self._on_message_callback(message)
            return None
        else:
            return self.read_queue.put(message)

    def ping(self, data: bytes=b'') -> None:
        """Send ping frame to the remote end.

        The data argument allows a small amount of data (up to 125
        bytes) to be sent as a part of the ping message. Note that not
        all websocket implementations expose this data to
        applications.

        Consider using the ``ping_interval`` argument to
        `websocket_connect` instead of sending pings manually.

        .. versionadded:: 5.1

        """
        data = utf8(data)
        if self.protocol is None:
            raise WebSocketClosedError()
        self.protocol.write_ping(data)

    def on_pong(self, data: bytes) -> None:
        pass

    def on_ping(self, data: bytes) -> None:
        pass

    def get_websocket_protocol(self) -> WebSocketProtocol:
        return WebSocketProtocol13(self, mask_outgoing=True, params=self.params)

    @property
    def selected_subprotocol(self) -> Optional[str]:
        """The subprotocol selected by the server.

        .. versionadded:: 5.1
        """
        return self.protocol.selected_subprotocol

    def log_exception(self, typ: 'Optional[Type[BaseException]]', value: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        assert typ is not None
        assert value is not None
        app_log.error('Uncaught exception %s', value, exc_info=(typ, value, tb))