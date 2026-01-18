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
class SSLIOStream(IOStream):
    """A utility class to write to and read from a non-blocking SSL socket.

    If the socket passed to the constructor is already connected,
    it should be wrapped with::

        ssl.SSLContext(...).wrap_socket(sock, do_handshake_on_connect=False, **kwargs)

    before constructing the `SSLIOStream`.  Unconnected sockets will be
    wrapped when `IOStream.connect` is finished.
    """
    socket = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """The ``ssl_options`` keyword argument may either be an
        `ssl.SSLContext` object or a dictionary of keywords arguments
        for `ssl.SSLContext.wrap_socket`
        """
        self._ssl_options = kwargs.pop('ssl_options', _client_ssl_defaults)
        super().__init__(*args, **kwargs)
        self._ssl_accepting = True
        self._handshake_reading = False
        self._handshake_writing = False
        self._server_hostname = None
        try:
            self.socket.getpeername()
        except socket.error:
            pass
        else:
            self._add_io_state(self.io_loop.WRITE)

    def reading(self) -> bool:
        return self._handshake_reading or super().reading()

    def writing(self) -> bool:
        return self._handshake_writing or super().writing()

    def _do_ssl_handshake(self) -> None:
        try:
            self._handshake_reading = False
            self._handshake_writing = False
            self.socket.do_handshake()
        except ssl.SSLError as err:
            if err.args[0] == ssl.SSL_ERROR_WANT_READ:
                self._handshake_reading = True
                return
            elif err.args[0] == ssl.SSL_ERROR_WANT_WRITE:
                self._handshake_writing = True
                return
            elif err.args[0] in (ssl.SSL_ERROR_EOF, ssl.SSL_ERROR_ZERO_RETURN):
                return self.close(exc_info=err)
            elif err.args[0] == ssl.SSL_ERROR_SSL:
                try:
                    peer = self.socket.getpeername()
                except Exception:
                    peer = '(not connected)'
                gen_log.warning('SSL Error on %s %s: %s', self.socket.fileno(), peer, err)
                return self.close(exc_info=err)
            raise
        except ssl.CertificateError as err:
            return self.close(exc_info=err)
        except socket.error as err:
            if self._is_connreset(err) or err.args[0] in (0, errno.EBADF, errno.ENOTCONN):
                return self.close(exc_info=err)
            raise
        except AttributeError as err:
            return self.close(exc_info=err)
        else:
            self._ssl_accepting = False
            assert ssl.HAS_SNI
            self._finish_ssl_connect()

    def _finish_ssl_connect(self) -> None:
        if self._ssl_connect_future is not None:
            future = self._ssl_connect_future
            self._ssl_connect_future = None
            future_set_result_unless_cancelled(future, self)

    def _handle_read(self) -> None:
        if self._ssl_accepting:
            self._do_ssl_handshake()
            return
        super()._handle_read()

    def _handle_write(self) -> None:
        if self._ssl_accepting:
            self._do_ssl_handshake()
            return
        super()._handle_write()

    def connect(self, address: Tuple, server_hostname: Optional[str]=None) -> 'Future[SSLIOStream]':
        self._server_hostname = server_hostname
        fut = super().connect(address)
        fut.add_done_callback(lambda f: f.exception())
        return self.wait_for_handshake()

    def _handle_connect(self) -> None:
        super()._handle_connect()
        if self.closed():
            return
        self.io_loop.remove_handler(self.socket)
        old_state = self._state
        assert old_state is not None
        self._state = None
        self.socket = ssl_wrap_socket(self.socket, self._ssl_options, server_hostname=self._server_hostname, do_handshake_on_connect=False, server_side=False)
        self._add_io_state(old_state)

    def wait_for_handshake(self) -> 'Future[SSLIOStream]':
        """Wait for the initial SSL handshake to complete.

        If a ``callback`` is given, it will be called with no
        arguments once the handshake is complete; otherwise this
        method returns a `.Future` which will resolve to the
        stream itself after the handshake is complete.

        Once the handshake is complete, information such as
        the peer's certificate and NPN/ALPN selections may be
        accessed on ``self.socket``.

        This method is intended for use on server-side streams
        or after using `IOStream.start_tls`; it should not be used
        with `IOStream.connect` (which already waits for the
        handshake to complete). It may only be called once per stream.

        .. versionadded:: 4.2

        .. versionchanged:: 6.0

           The ``callback`` argument was removed. Use the returned
           `.Future` instead.

        """
        if self._ssl_connect_future is not None:
            raise RuntimeError('Already waiting')
        future = self._ssl_connect_future = Future()
        if not self._ssl_accepting:
            self._finish_ssl_connect()
        return future

    def write_to_fd(self, data: memoryview) -> int:
        if len(data) >> 30:
            data = memoryview(data)[:1 << 30]
        try:
            return self.socket.send(data)
        except ssl.SSLError as e:
            if e.args[0] == ssl.SSL_ERROR_WANT_WRITE:
                return 0
            raise
        finally:
            del data

    def read_from_fd(self, buf: Union[bytearray, memoryview]) -> Optional[int]:
        try:
            if self._ssl_accepting:
                return None
            if len(buf) >> 30:
                buf = memoryview(buf)[:1 << 30]
            try:
                return self.socket.recv_into(buf, len(buf))
            except ssl.SSLError as e:
                if e.args[0] == ssl.SSL_ERROR_WANT_READ:
                    return None
                else:
                    raise
            except BlockingIOError:
                return None
        finally:
            del buf

    def _is_connreset(self, e: BaseException) -> bool:
        if isinstance(e, ssl.SSLError) and e.args[0] == ssl.SSL_ERROR_EOF:
            return True
        return super()._is_connreset(e)