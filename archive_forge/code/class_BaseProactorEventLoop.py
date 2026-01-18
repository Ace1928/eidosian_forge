import io
import os
import socket
import warnings
import signal
import threading
import collections
from . import base_events
from . import constants
from . import futures
from . import exceptions
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
class BaseProactorEventLoop(base_events.BaseEventLoop):

    def __init__(self, proactor):
        super().__init__()
        logger.debug('Using proactor: %s', proactor.__class__.__name__)
        self._proactor = proactor
        self._selector = proactor
        self._self_reading_future = None
        self._accept_futures = {}
        proactor.set_loop(self)
        self._make_self_pipe()
        if threading.current_thread() is threading.main_thread():
            signal.set_wakeup_fd(self._csock.fileno())

    def _make_socket_transport(self, sock, protocol, waiter=None, extra=None, server=None):
        return _ProactorSocketTransport(self, sock, protocol, waiter, extra, server)

    def _make_ssl_transport(self, rawsock, protocol, sslcontext, waiter=None, *, server_side=False, server_hostname=None, extra=None, server=None, ssl_handshake_timeout=None, ssl_shutdown_timeout=None):
        ssl_protocol = sslproto.SSLProtocol(self, protocol, sslcontext, waiter, server_side, server_hostname, ssl_handshake_timeout=ssl_handshake_timeout, ssl_shutdown_timeout=ssl_shutdown_timeout)
        _ProactorSocketTransport(self, rawsock, ssl_protocol, extra=extra, server=server)
        return ssl_protocol._app_transport

    def _make_datagram_transport(self, sock, protocol, address=None, waiter=None, extra=None):
        return _ProactorDatagramTransport(self, sock, protocol, address, waiter, extra)

    def _make_duplex_pipe_transport(self, sock, protocol, waiter=None, extra=None):
        return _ProactorDuplexPipeTransport(self, sock, protocol, waiter, extra)

    def _make_read_pipe_transport(self, sock, protocol, waiter=None, extra=None):
        return _ProactorReadPipeTransport(self, sock, protocol, waiter, extra)

    def _make_write_pipe_transport(self, sock, protocol, waiter=None, extra=None):
        return _ProactorWritePipeTransport(self, sock, protocol, waiter, extra)

    def close(self):
        if self.is_running():
            raise RuntimeError('Cannot close a running event loop')
        if self.is_closed():
            return
        if threading.current_thread() is threading.main_thread():
            signal.set_wakeup_fd(-1)
        self._stop_accept_futures()
        self._close_self_pipe()
        self._proactor.close()
        self._proactor = None
        self._selector = None
        super().close()

    async def sock_recv(self, sock, n):
        return await self._proactor.recv(sock, n)

    async def sock_recv_into(self, sock, buf):
        return await self._proactor.recv_into(sock, buf)

    async def sock_recvfrom(self, sock, bufsize):
        return await self._proactor.recvfrom(sock, bufsize)

    async def sock_recvfrom_into(self, sock, buf, nbytes=0):
        if not nbytes:
            nbytes = len(buf)
        return await self._proactor.recvfrom_into(sock, buf, nbytes)

    async def sock_sendall(self, sock, data):
        return await self._proactor.send(sock, data)

    async def sock_sendto(self, sock, data, address):
        return await self._proactor.sendto(sock, data, 0, address)

    async def sock_connect(self, sock, address):
        return await self._proactor.connect(sock, address)

    async def sock_accept(self, sock):
        return await self._proactor.accept(sock)

    async def _sock_sendfile_native(self, sock, file, offset, count):
        try:
            fileno = file.fileno()
        except (AttributeError, io.UnsupportedOperation) as err:
            raise exceptions.SendfileNotAvailableError('not a regular file')
        try:
            fsize = os.fstat(fileno).st_size
        except OSError:
            raise exceptions.SendfileNotAvailableError('not a regular file')
        blocksize = count if count else fsize
        if not blocksize:
            return 0
        blocksize = min(blocksize, 4294967295)
        end_pos = min(offset + count, fsize) if count else fsize
        offset = min(offset, fsize)
        total_sent = 0
        try:
            while True:
                blocksize = min(end_pos - offset, blocksize)
                if blocksize <= 0:
                    return total_sent
                await self._proactor.sendfile(sock, file, offset, blocksize)
                offset += blocksize
                total_sent += blocksize
        finally:
            if total_sent > 0:
                file.seek(offset)

    async def _sendfile_native(self, transp, file, offset, count):
        resume_reading = transp.is_reading()
        transp.pause_reading()
        await transp._make_empty_waiter()
        try:
            return await self.sock_sendfile(transp._sock, file, offset, count, fallback=False)
        finally:
            transp._reset_empty_waiter()
            if resume_reading:
                transp.resume_reading()

    def _close_self_pipe(self):
        if self._self_reading_future is not None:
            self._self_reading_future.cancel()
            self._self_reading_future = None
        self._ssock.close()
        self._ssock = None
        self._csock.close()
        self._csock = None
        self._internal_fds -= 1

    def _make_self_pipe(self):
        self._ssock, self._csock = socket.socketpair()
        self._ssock.setblocking(False)
        self._csock.setblocking(False)
        self._internal_fds += 1

    def _loop_self_reading(self, f=None):
        try:
            if f is not None:
                f.result()
            if self._self_reading_future is not f:
                return
            f = self._proactor.recv(self._ssock, 4096)
        except exceptions.CancelledError:
            return
        except (SystemExit, KeyboardInterrupt):
            raise
        except BaseException as exc:
            self.call_exception_handler({'message': 'Error on reading from the event loop self pipe', 'exception': exc, 'loop': self})
        else:
            self._self_reading_future = f
            f.add_done_callback(self._loop_self_reading)

    def _write_to_self(self):
        csock = self._csock
        if csock is None:
            return
        try:
            csock.send(b'\x00')
        except OSError:
            if self._debug:
                logger.debug('Fail to write a null byte into the self-pipe socket', exc_info=True)

    def _start_serving(self, protocol_factory, sock, sslcontext=None, server=None, backlog=100, ssl_handshake_timeout=None, ssl_shutdown_timeout=None):

        def loop(f=None):
            try:
                if f is not None:
                    conn, addr = f.result()
                    if self._debug:
                        logger.debug('%r got a new connection from %r: %r', server, addr, conn)
                    protocol = protocol_factory()
                    if sslcontext is not None:
                        self._make_ssl_transport(conn, protocol, sslcontext, server_side=True, extra={'peername': addr}, server=server, ssl_handshake_timeout=ssl_handshake_timeout, ssl_shutdown_timeout=ssl_shutdown_timeout)
                    else:
                        self._make_socket_transport(conn, protocol, extra={'peername': addr}, server=server)
                if self.is_closed():
                    return
                f = self._proactor.accept(sock)
            except OSError as exc:
                if sock.fileno() != -1:
                    self.call_exception_handler({'message': 'Accept failed on a socket', 'exception': exc, 'socket': trsock.TransportSocket(sock)})
                    sock.close()
                elif self._debug:
                    logger.debug('Accept failed on socket %r', sock, exc_info=True)
            except exceptions.CancelledError:
                sock.close()
            else:
                self._accept_futures[sock.fileno()] = f
                f.add_done_callback(loop)
        self.call_soon(loop)

    def _process_events(self, event_list):
        pass

    def _stop_accept_futures(self):
        for future in self._accept_futures.values():
            future.cancel()
        self._accept_futures.clear()

    def _stop_serving(self, sock):
        future = self._accept_futures.pop(sock.fileno(), None)
        if future:
            future.cancel()
        self._proactor._stop_serving(sock)
        sock.close()