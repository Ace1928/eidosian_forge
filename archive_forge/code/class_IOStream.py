import asyncio
import collections
import errno
import io
import numbers
import os
import socket
import ssl
import sys
import re
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import ioloop
from tornado.log import gen_log
from tornado.netutil import ssl_wrap_socket, _client_ssl_defaults, _server_ssl_defaults
from tornado.util import errno_from_exception
import typing
from typing import (
from types import TracebackType
class IOStream(BaseIOStream):
    """Socket-based `IOStream` implementation.

    This class supports the read and write methods from `BaseIOStream`
    plus a `connect` method.

    The ``socket`` parameter may either be connected or unconnected.
    For server operations the socket is the result of calling
    `socket.accept <socket.socket.accept>`.  For client operations the
    socket is created with `socket.socket`, and may either be
    connected before passing it to the `IOStream` or connected with
    `IOStream.connect`.

    A very simple (and broken) HTTP client using this class:

    .. testcode::

        import socket
        import tornado

        async def main():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
            stream = tornado.iostream.IOStream(s)
            await stream.connect(("friendfeed.com", 80))
            await stream.write(b"GET / HTTP/1.0\\r\\nHost: friendfeed.com\\r\\n\\r\\n")
            header_data = await stream.read_until(b"\\r\\n\\r\\n")
            headers = {}
            for line in header_data.split(b"\\r\\n"):
                parts = line.split(b":")
                if len(parts) == 2:
                    headers[parts[0].strip()] = parts[1].strip()
            body_data = await stream.read_bytes(int(headers[b"Content-Length"]))
            print(body_data)
            stream.close()

        if __name__ == '__main__':
            asyncio.run(main())

    .. testoutput::
       :hide:

    """

    def __init__(self, socket: socket.socket, *args: Any, **kwargs: Any) -> None:
        self.socket = socket
        self.socket.setblocking(False)
        super().__init__(*args, **kwargs)

    def fileno(self) -> Union[int, ioloop._Selectable]:
        return self.socket

    def close_fd(self) -> None:
        self.socket.close()
        self.socket = None

    def get_fd_error(self) -> Optional[Exception]:
        errno = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        return socket.error(errno, os.strerror(errno))

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        try:
            return self.socket.recv_into(buf, len(buf))
        except BlockingIOError:
            return None
        finally:
            del buf

    def write_to_fd(self, data: memoryview) -> int:
        try:
            return self.socket.send(data)
        finally:
            del data

    def connect(self: _IOStreamType, address: Any, server_hostname: Optional[str]=None) -> 'Future[_IOStreamType]':
        """Connects the socket to a remote address without blocking.

        May only be called if the socket passed to the constructor was
        not previously connected.  The address parameter is in the
        same format as for `socket.connect <socket.socket.connect>` for
        the type of socket passed to the IOStream constructor,
        e.g. an ``(ip, port)`` tuple.  Hostnames are accepted here,
        but will be resolved synchronously and block the IOLoop.
        If you have a hostname instead of an IP address, the `.TCPClient`
        class is recommended instead of calling this method directly.
        `.TCPClient` will do asynchronous DNS resolution and handle
        both IPv4 and IPv6.

        If ``callback`` is specified, it will be called with no
        arguments when the connection is completed; if not this method
        returns a `.Future` (whose result after a successful
        connection will be the stream itself).

        In SSL mode, the ``server_hostname`` parameter will be used
        for certificate validation (unless disabled in the
        ``ssl_options``) and SNI (if supported; requires Python
        2.7.9+).

        Note that it is safe to call `IOStream.write
        <BaseIOStream.write>` while the connection is pending, in
        which case the data will be written as soon as the connection
        is ready.  Calling `IOStream` read methods before the socket is
        connected works on some platforms but is non-portable.

        .. versionchanged:: 4.0
            If no callback is given, returns a `.Future`.

        .. versionchanged:: 4.2
           SSL certificates are validated by default; pass
           ``ssl_options=dict(cert_reqs=ssl.CERT_NONE)`` or a
           suitably-configured `ssl.SSLContext` to the
           `SSLIOStream` constructor to disable.

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        self._connecting = True
        future = Future()
        self._connect_future = typing.cast('Future[IOStream]', future)
        try:
            self.socket.connect(address)
        except BlockingIOError:
            pass
        except socket.error as e:
            if future is None:
                gen_log.warning('Connect error on fd %s: %s', self.socket.fileno(), e)
            self.close(exc_info=e)
            return future
        self._add_io_state(self.io_loop.WRITE)
        return future

    def start_tls(self, server_side: bool, ssl_options: Optional[Union[Dict[str, Any], ssl.SSLContext]]=None, server_hostname: Optional[str]=None) -> Awaitable['SSLIOStream']:
        """Convert this `IOStream` to an `SSLIOStream`.

        This enables protocols that begin in clear-text mode and
        switch to SSL after some initial negotiation (such as the
        ``STARTTLS`` extension to SMTP and IMAP).

        This method cannot be used if there are outstanding reads
        or writes on the stream, or if there is any data in the
        IOStream's buffer (data in the operating system's socket
        buffer is allowed).  This means it must generally be used
        immediately after reading or writing the last clear-text
        data.  It can also be used immediately after connecting,
        before any reads or writes.

        The ``ssl_options`` argument may be either an `ssl.SSLContext`
        object or a dictionary of keyword arguments for the
        `ssl.SSLContext.wrap_socket` function.  The ``server_hostname`` argument
        will be used for certificate validation unless disabled
        in the ``ssl_options``.

        This method returns a `.Future` whose result is the new
        `SSLIOStream`.  After this method has been called,
        any other operation on the original stream is undefined.

        If a close callback is defined on this stream, it will be
        transferred to the new stream.

        .. versionadded:: 4.0

        .. versionchanged:: 4.2
           SSL certificates are validated by default; pass
           ``ssl_options=dict(cert_reqs=ssl.CERT_NONE)`` or a
           suitably-configured `ssl.SSLContext` to disable.
        """
        if self._read_future or self._write_futures or self._connect_future or self._closed or self._read_buffer or self._write_buffer:
            raise ValueError('IOStream is not idle; cannot convert to SSL')
        if ssl_options is None:
            if server_side:
                ssl_options = _server_ssl_defaults
            else:
                ssl_options = _client_ssl_defaults
        socket = self.socket
        self.io_loop.remove_handler(socket)
        self.socket = None
        socket = ssl_wrap_socket(socket, ssl_options, server_hostname=server_hostname, server_side=server_side, do_handshake_on_connect=False)
        orig_close_callback = self._close_callback
        self._close_callback = None
        future = Future()
        ssl_stream = SSLIOStream(socket, ssl_options=ssl_options)
        ssl_stream.set_close_callback(orig_close_callback)
        ssl_stream._ssl_connect_future = future
        ssl_stream.max_buffer_size = self.max_buffer_size
        ssl_stream.read_chunk_size = self.read_chunk_size
        return future

    def _handle_connect(self) -> None:
        try:
            err = self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
        except socket.error as e:
            if errno_from_exception(e) == errno.ENOPROTOOPT:
                err = 0
        if err != 0:
            self.error = socket.error(err, os.strerror(err))
            if self._connect_future is None:
                gen_log.warning('Connect error on fd %s: %s', self.socket.fileno(), errno.errorcode[err])
            self.close()
            return
        if self._connect_future is not None:
            future = self._connect_future
            self._connect_future = None
            future_set_result_unless_cancelled(future, self)
        self._connecting = False

    def set_nodelay(self, value: bool) -> None:
        if self.socket is not None and self.socket.family in (socket.AF_INET, socket.AF_INET6):
            try:
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1 if value else 0)
            except socket.error as e:
                if e.errno != errno.EINVAL and (not self._is_connreset(e)):
                    raise