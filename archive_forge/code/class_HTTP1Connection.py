import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
class HTTP1Connection(httputil.HTTPConnection):
    """Implements the HTTP/1.x protocol.

    This class can be on its own for clients, or via `HTTP1ServerConnection`
    for servers.
    """

    def __init__(self, stream: iostream.IOStream, is_client: bool, params: Optional[HTTP1ConnectionParameters]=None, context: Optional[object]=None) -> None:
        """
        :arg stream: an `.IOStream`
        :arg bool is_client: client or server
        :arg params: a `.HTTP1ConnectionParameters` instance or ``None``
        :arg context: an opaque application-defined object that can be accessed
            as ``connection.context``.
        """
        self.is_client = is_client
        self.stream = stream
        if params is None:
            params = HTTP1ConnectionParameters()
        self.params = params
        self.context = context
        self.no_keep_alive = params.no_keep_alive
        self._max_body_size = self.params.max_body_size if self.params.max_body_size is not None else self.stream.max_buffer_size
        self._body_timeout = self.params.body_timeout
        self._write_finished = False
        self._read_finished = False
        self._finish_future = Future()
        self._disconnect_on_finish = False
        self._clear_callbacks()
        self._request_start_line = None
        self._response_start_line = None
        self._request_headers = None
        self._chunking_output = False
        self._expected_content_remaining = None
        self._pending_write = None

    def read_response(self, delegate: httputil.HTTPMessageDelegate) -> Awaitable[bool]:
        """Read a single HTTP response.

        Typical client-mode usage is to write a request using `write_headers`,
        `write`, and `finish`, and then call ``read_response``.

        :arg delegate: a `.HTTPMessageDelegate`

        Returns a `.Future` that resolves to a bool after the full response has
        been read. The result is true if the stream is still open.
        """
        if self.params.decompress:
            delegate = _GzipMessageDelegate(delegate, self.params.chunk_size)
        return self._read_message(delegate)

    async def _read_message(self, delegate: httputil.HTTPMessageDelegate) -> bool:
        need_delegate_close = False
        try:
            header_future = self.stream.read_until_regex(b'\r?\n\r?\n', max_bytes=self.params.max_header_size)
            if self.params.header_timeout is None:
                header_data = await header_future
            else:
                try:
                    header_data = await gen.with_timeout(self.stream.io_loop.time() + self.params.header_timeout, header_future, quiet_exceptions=iostream.StreamClosedError)
                except gen.TimeoutError:
                    self.close()
                    return False
            start_line_str, headers = self._parse_headers(header_data)
            if self.is_client:
                resp_start_line = httputil.parse_response_start_line(start_line_str)
                self._response_start_line = resp_start_line
                start_line = resp_start_line
                self._disconnect_on_finish = False
            else:
                req_start_line = httputil.parse_request_start_line(start_line_str)
                self._request_start_line = req_start_line
                self._request_headers = headers
                start_line = req_start_line
                self._disconnect_on_finish = not self._can_keep_alive(req_start_line, headers)
            need_delegate_close = True
            with _ExceptionLoggingContext(app_log):
                header_recv_future = delegate.headers_received(start_line, headers)
                if header_recv_future is not None:
                    await header_recv_future
            if self.stream is None:
                need_delegate_close = False
                return False
            skip_body = False
            if self.is_client:
                assert isinstance(start_line, httputil.ResponseStartLine)
                if self._request_start_line is not None and self._request_start_line.method == 'HEAD':
                    skip_body = True
                code = start_line.code
                if code == 304:
                    skip_body = True
                if 100 <= code < 200:
                    if 'Content-Length' in headers or 'Transfer-Encoding' in headers:
                        raise httputil.HTTPInputError('Response code %d cannot have body' % code)
                    await self._read_message(delegate)
            elif headers.get('Expect') == '100-continue' and (not self._write_finished):
                self.stream.write(b'HTTP/1.1 100 (Continue)\r\n\r\n')
            if not skip_body:
                body_future = self._read_body(resp_start_line.code if self.is_client else 0, headers, delegate)
                if body_future is not None:
                    if self._body_timeout is None:
                        await body_future
                    else:
                        try:
                            await gen.with_timeout(self.stream.io_loop.time() + self._body_timeout, body_future, quiet_exceptions=iostream.StreamClosedError)
                        except gen.TimeoutError:
                            gen_log.info('Timeout reading body from %s', self.context)
                            self.stream.close()
                            return False
            self._read_finished = True
            if not self._write_finished or self.is_client:
                need_delegate_close = False
                with _ExceptionLoggingContext(app_log):
                    delegate.finish()
            if not self._finish_future.done() and self.stream is not None and (not self.stream.closed()):
                self.stream.set_close_callback(self._on_connection_close)
                await self._finish_future
            if self.is_client and self._disconnect_on_finish:
                self.close()
            if self.stream is None:
                return False
        except httputil.HTTPInputError as e:
            gen_log.info('Malformed HTTP message from %s: %s', self.context, e)
            if not self.is_client:
                await self.stream.write(b'HTTP/1.1 400 Bad Request\r\n\r\n')
            self.close()
            return False
        finally:
            if need_delegate_close:
                with _ExceptionLoggingContext(app_log):
                    delegate.on_connection_close()
            header_future = None
            self._clear_callbacks()
        return True

    def _clear_callbacks(self) -> None:
        """Clears the callback attributes.

        This allows the request handler to be garbage collected more
        quickly in CPython by breaking up reference cycles.
        """
        self._write_callback = None
        self._write_future = None
        self._close_callback = None
        if self.stream is not None:
            self.stream.set_close_callback(None)

    def set_close_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Sets a callback that will be run when the connection is closed.

        Note that this callback is slightly different from
        `.HTTPMessageDelegate.on_connection_close`: The
        `.HTTPMessageDelegate` method is called when the connection is
        closed while receiving a message. This callback is used when
        there is not an active delegate (for example, on the server
        side this callback is used if the client closes the connection
        after sending its request but before receiving all the
        response.
        """
        self._close_callback = callback

    def _on_connection_close(self) -> None:
        if self._close_callback is not None:
            callback = self._close_callback
            self._close_callback = None
            callback()
        if not self._finish_future.done():
            future_set_result_unless_cancelled(self._finish_future, None)
        self._clear_callbacks()

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
        self._clear_callbacks()
        if not self._finish_future.done():
            future_set_result_unless_cancelled(self._finish_future, None)

    def detach(self) -> iostream.IOStream:
        """Take control of the underlying stream.

        Returns the underlying `.IOStream` object and stops all further
        HTTP processing.  May only be called during
        `.HTTPMessageDelegate.headers_received`.  Intended for implementing
        protocols like websockets that tunnel over an HTTP handshake.
        """
        self._clear_callbacks()
        stream = self.stream
        self.stream = None
        if not self._finish_future.done():
            future_set_result_unless_cancelled(self._finish_future, None)
        return stream

    def set_body_timeout(self, timeout: float) -> None:
        """Sets the body timeout for a single request.

        Overrides the value from `.HTTP1ConnectionParameters`.
        """
        self._body_timeout = timeout

    def set_max_body_size(self, max_body_size: int) -> None:
        """Sets the body size limit for a single request.

        Overrides the value from `.HTTP1ConnectionParameters`.
        """
        self._max_body_size = max_body_size

    def write_headers(self, start_line: Union[httputil.RequestStartLine, httputil.ResponseStartLine], headers: httputil.HTTPHeaders, chunk: Optional[bytes]=None) -> 'Future[None]':
        """Implements `.HTTPConnection.write_headers`."""
        lines = []
        if self.is_client:
            assert isinstance(start_line, httputil.RequestStartLine)
            self._request_start_line = start_line
            lines.append(utf8('%s %s HTTP/1.1' % (start_line[0], start_line[1])))
            self._chunking_output = start_line.method in ('POST', 'PUT', 'PATCH') and 'Content-Length' not in headers and ('Transfer-Encoding' not in headers or headers['Transfer-Encoding'] == 'chunked')
        else:
            assert isinstance(start_line, httputil.ResponseStartLine)
            assert self._request_start_line is not None
            assert self._request_headers is not None
            self._response_start_line = start_line
            lines.append(utf8('HTTP/1.1 %d %s' % (start_line[1], start_line[2])))
            self._chunking_output = self._request_start_line.version == 'HTTP/1.1' and self._request_start_line.method != 'HEAD' and (start_line.code not in (204, 304)) and (start_line.code < 100 or start_line.code >= 200) and ('Content-Length' not in headers) and ('Transfer-Encoding' not in headers)
            if self._request_start_line.version == 'HTTP/1.1' and self._disconnect_on_finish:
                headers['Connection'] = 'close'
            if self._request_start_line.version == 'HTTP/1.0' and self._request_headers.get('Connection', '').lower() == 'keep-alive':
                headers['Connection'] = 'Keep-Alive'
        if self._chunking_output:
            headers['Transfer-Encoding'] = 'chunked'
        if not self.is_client and (self._request_start_line.method == 'HEAD' or cast(httputil.ResponseStartLine, start_line).code == 304):
            self._expected_content_remaining = 0
        elif 'Content-Length' in headers:
            self._expected_content_remaining = parse_int(headers['Content-Length'])
        else:
            self._expected_content_remaining = None
        header_lines = (native_str(n) + ': ' + native_str(v) for n, v in headers.get_all())
        lines.extend((line.encode('latin1') for line in header_lines))
        for line in lines:
            if b'\n' in line:
                raise ValueError('Newline in header: ' + repr(line))
        future = None
        if self.stream.closed():
            future = self._write_future = Future()
            future.set_exception(iostream.StreamClosedError())
            future.exception()
        else:
            future = self._write_future = Future()
            data = b'\r\n'.join(lines) + b'\r\n\r\n'
            if chunk:
                data += self._format_chunk(chunk)
            self._pending_write = self.stream.write(data)
            future_add_done_callback(self._pending_write, self._on_write_complete)
        return future

    def _format_chunk(self, chunk: bytes) -> bytes:
        if self._expected_content_remaining is not None:
            self._expected_content_remaining -= len(chunk)
            if self._expected_content_remaining < 0:
                self.stream.close()
                raise httputil.HTTPOutputError('Tried to write more data than Content-Length')
        if self._chunking_output and chunk:
            return utf8('%x' % len(chunk)) + b'\r\n' + chunk + b'\r\n'
        else:
            return chunk

    def write(self, chunk: bytes) -> 'Future[None]':
        """Implements `.HTTPConnection.write`.

        For backwards compatibility it is allowed but deprecated to
        skip `write_headers` and instead call `write()` with a
        pre-encoded header block.
        """
        future = None
        if self.stream.closed():
            future = self._write_future = Future()
            self._write_future.set_exception(iostream.StreamClosedError())
            self._write_future.exception()
        else:
            future = self._write_future = Future()
            self._pending_write = self.stream.write(self._format_chunk(chunk))
            future_add_done_callback(self._pending_write, self._on_write_complete)
        return future

    def finish(self) -> None:
        """Implements `.HTTPConnection.finish`."""
        if self._expected_content_remaining is not None and self._expected_content_remaining != 0 and (not self.stream.closed()):
            self.stream.close()
            raise httputil.HTTPOutputError('Tried to write %d bytes less than Content-Length' % self._expected_content_remaining)
        if self._chunking_output:
            if not self.stream.closed():
                self._pending_write = self.stream.write(b'0\r\n\r\n')
                self._pending_write.add_done_callback(self._on_write_complete)
        self._write_finished = True
        if not self._read_finished:
            self._disconnect_on_finish = True
        self.stream.set_nodelay(True)
        if self._pending_write is None:
            self._finish_request(None)
        else:
            future_add_done_callback(self._pending_write, self._finish_request)

    def _on_write_complete(self, future: 'Future[None]') -> None:
        exc = future.exception()
        if exc is not None and (not isinstance(exc, iostream.StreamClosedError)):
            future.result()
        if self._write_callback is not None:
            callback = self._write_callback
            self._write_callback = None
            self.stream.io_loop.add_callback(callback)
        if self._write_future is not None:
            future = self._write_future
            self._write_future = None
            future_set_result_unless_cancelled(future, None)

    def _can_keep_alive(self, start_line: httputil.RequestStartLine, headers: httputil.HTTPHeaders) -> bool:
        if self.params.no_keep_alive:
            return False
        connection_header = headers.get('Connection')
        if connection_header is not None:
            connection_header = connection_header.lower()
        if start_line.version == 'HTTP/1.1':
            return connection_header != 'close'
        elif 'Content-Length' in headers or headers.get('Transfer-Encoding', '').lower() == 'chunked' or getattr(start_line, 'method', None) in ('HEAD', 'GET'):
            return connection_header == 'keep-alive'
        return False

    def _finish_request(self, future: 'Optional[Future[None]]') -> None:
        self._clear_callbacks()
        if not self.is_client and self._disconnect_on_finish:
            self.close()
            return
        self.stream.set_nodelay(False)
        if not self._finish_future.done():
            future_set_result_unless_cancelled(self._finish_future, None)

    def _parse_headers(self, data: bytes) -> Tuple[str, httputil.HTTPHeaders]:
        data_str = native_str(data.decode('latin1')).lstrip('\r\n')
        eol = data_str.find('\n')
        start_line = data_str[:eol].rstrip('\r')
        headers = httputil.HTTPHeaders.parse(data_str[eol:])
        return (start_line, headers)

    def _read_body(self, code: int, headers: httputil.HTTPHeaders, delegate: httputil.HTTPMessageDelegate) -> Optional[Awaitable[None]]:
        if 'Content-Length' in headers:
            if 'Transfer-Encoding' in headers:
                raise httputil.HTTPInputError('Response with both Transfer-Encoding and Content-Length')
            if ',' in headers['Content-Length']:
                pieces = re.split(',\\s*', headers['Content-Length'])
                if any((i != pieces[0] for i in pieces)):
                    raise httputil.HTTPInputError('Multiple unequal Content-Lengths: %r' % headers['Content-Length'])
                headers['Content-Length'] = pieces[0]
            try:
                content_length: Optional[int] = parse_int(headers['Content-Length'])
            except ValueError:
                raise httputil.HTTPInputError('Only integer Content-Length is allowed: %s' % headers['Content-Length'])
            if cast(int, content_length) > self._max_body_size:
                raise httputil.HTTPInputError('Content-Length too long')
        else:
            content_length = None
        if code == 204:
            if 'Transfer-Encoding' in headers or content_length not in (None, 0):
                raise httputil.HTTPInputError('Response with code %d should not have body' % code)
            content_length = 0
        if content_length is not None:
            return self._read_fixed_body(content_length, delegate)
        if headers.get('Transfer-Encoding', '').lower() == 'chunked':
            return self._read_chunked_body(delegate)
        if self.is_client:
            return self._read_body_until_close(delegate)
        return None

    async def _read_fixed_body(self, content_length: int, delegate: httputil.HTTPMessageDelegate) -> None:
        while content_length > 0:
            body = await self.stream.read_bytes(min(self.params.chunk_size, content_length), partial=True)
            content_length -= len(body)
            if not self._write_finished or self.is_client:
                with _ExceptionLoggingContext(app_log):
                    ret = delegate.data_received(body)
                    if ret is not None:
                        await ret

    async def _read_chunked_body(self, delegate: httputil.HTTPMessageDelegate) -> None:
        total_size = 0
        while True:
            chunk_len_str = await self.stream.read_until(b'\r\n', max_bytes=64)
            try:
                chunk_len = parse_hex_int(native_str(chunk_len_str[:-2]))
            except ValueError:
                raise httputil.HTTPInputError('invalid chunk size')
            if chunk_len == 0:
                crlf = await self.stream.read_bytes(2)
                if crlf != b'\r\n':
                    raise httputil.HTTPInputError('improperly terminated chunked request')
                return
            total_size += chunk_len
            if total_size > self._max_body_size:
                raise httputil.HTTPInputError('chunked body too large')
            bytes_to_read = chunk_len
            while bytes_to_read:
                chunk = await self.stream.read_bytes(min(bytes_to_read, self.params.chunk_size), partial=True)
                bytes_to_read -= len(chunk)
                if not self._write_finished or self.is_client:
                    with _ExceptionLoggingContext(app_log):
                        ret = delegate.data_received(chunk)
                        if ret is not None:
                            await ret
            crlf = await self.stream.read_bytes(2)
            assert crlf == b'\r\n'

    async def _read_body_until_close(self, delegate: httputil.HTTPMessageDelegate) -> None:
        body = await self.stream.read_until_close()
        if not self._write_finished or self.is_client:
            with _ExceptionLoggingContext(app_log):
                ret = delegate.data_received(body)
                if ret is not None:
                    await ret