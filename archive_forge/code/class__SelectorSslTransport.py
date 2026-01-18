import collections
import errno
import functools
import socket
import sys
import warnings
from . import base_events
from . import constants
from . import events
from . import futures
from . import selectors
from . import transports
from . import sslproto
from .coroutines import coroutine
from .log import logger
class _SelectorSslTransport(_SelectorTransport):
    _buffer_factory = bytearray

    def __init__(self, loop, rawsock, protocol, sslcontext, waiter=None, server_side=False, server_hostname=None, extra=None, server=None):
        if ssl is None:
            raise RuntimeError('stdlib ssl module not available')
        if not sslcontext:
            sslcontext = sslproto._create_transport_context(server_side, server_hostname)
        wrap_kwargs = {'server_side': server_side, 'do_handshake_on_connect': False}
        if server_hostname and (not server_side):
            wrap_kwargs['server_hostname'] = server_hostname
        sslsock = sslcontext.wrap_socket(rawsock, **wrap_kwargs)
        super().__init__(loop, sslsock, protocol, extra, server)
        self._protocol_connected = False
        self._server_hostname = server_hostname
        self._waiter = waiter
        self._sslcontext = sslcontext
        self._paused = False
        self._extra.update(sslcontext=sslcontext)
        if self._loop.get_debug():
            logger.debug('%r starts SSL handshake', self)
            start_time = self._loop.time()
        else:
            start_time = None
        self._on_handshake(start_time)

    def _wakeup_waiter(self, exc=None):
        if self._waiter is None:
            return
        if not self._waiter.cancelled():
            if exc is not None:
                self._waiter.set_exception(exc)
            else:
                self._waiter.set_result(None)
        self._waiter = None

    def _on_handshake(self, start_time):
        try:
            self._sock.do_handshake()
        except ssl.SSLWantReadError:
            self._loop.add_reader(self._sock_fd, self._on_handshake, start_time)
            return
        except ssl.SSLWantWriteError:
            self._loop.add_writer(self._sock_fd, self._on_handshake, start_time)
            return
        except BaseException as exc:
            if self._loop.get_debug():
                logger.warning('%r: SSL handshake failed', self, exc_info=True)
            self._loop.remove_reader(self._sock_fd)
            self._loop.remove_writer(self._sock_fd)
            self._sock.close()
            self._wakeup_waiter(exc)
            if isinstance(exc, Exception):
                return
            else:
                raise
        self._loop.remove_reader(self._sock_fd)
        self._loop.remove_writer(self._sock_fd)
        peercert = self._sock.getpeercert()
        if not hasattr(self._sslcontext, 'check_hostname'):
            if self._server_hostname and self._sslcontext.verify_mode != ssl.CERT_NONE:
                try:
                    ssl.match_hostname(peercert, self._server_hostname)
                except Exception as exc:
                    if self._loop.get_debug():
                        logger.warning('%r: SSL handshake failed on matching the hostname', self, exc_info=True)
                    self._sock.close()
                    self._wakeup_waiter(exc)
                    return
        self._extra.update(peercert=peercert, cipher=self._sock.cipher(), compression=self._sock.compression())
        self._read_wants_write = False
        self._write_wants_read = False
        self._loop.add_reader(self._sock_fd, self._read_ready)
        self._protocol_connected = True
        self._loop.call_soon(self._protocol.connection_made, self)
        self._loop.call_soon(self._wakeup_waiter)
        if self._loop.get_debug():
            dt = self._loop.time() - start_time
            logger.debug('%r: SSL handshake took %.1f ms', self, dt * 1000.0)

    def pause_reading(self):
        if self._closing:
            raise RuntimeError('Cannot pause_reading() when closing')
        if self._paused:
            raise RuntimeError('Already paused')
        self._paused = True
        self._loop.remove_reader(self._sock_fd)
        if self._loop.get_debug():
            logger.debug('%r pauses reading', self)

    def resume_reading(self):
        if not self._paused:
            raise RuntimeError('Not paused')
        self._paused = False
        if self._closing:
            return
        self._loop.add_reader(self._sock_fd, self._read_ready)
        if self._loop.get_debug():
            logger.debug('%r resumes reading', self)

    def _read_ready(self):
        if self._write_wants_read:
            self._write_wants_read = False
            self._write_ready()
            if self._buffer:
                self._loop.add_writer(self._sock_fd, self._write_ready)
        try:
            data = self._sock.recv(self.max_size)
        except (BlockingIOError, InterruptedError, ssl.SSLWantReadError):
            pass
        except ssl.SSLWantWriteError:
            self._read_wants_write = True
            self._loop.remove_reader(self._sock_fd)
            self._loop.add_writer(self._sock_fd, self._write_ready)
        except Exception as exc:
            self._fatal_error(exc, 'Fatal read error on SSL transport')
        else:
            if data:
                self._protocol.data_received(data)
            else:
                try:
                    if self._loop.get_debug():
                        logger.debug('%r received EOF', self)
                    keep_open = self._protocol.eof_received()
                    if keep_open:
                        logger.warning('returning true from eof_received() has no effect when using ssl')
                finally:
                    self.close()

    def _write_ready(self):
        if self._read_wants_write:
            self._read_wants_write = False
            self._read_ready()
            if not (self._paused or self._closing):
                self._loop.add_reader(self._sock_fd, self._read_ready)
        if self._buffer:
            try:
                n = self._sock.send(self._buffer)
            except (BlockingIOError, InterruptedError, ssl.SSLWantWriteError):
                n = 0
            except ssl.SSLWantReadError:
                n = 0
                self._loop.remove_writer(self._sock_fd)
                self._write_wants_read = True
            except Exception as exc:
                self._loop.remove_writer(self._sock_fd)
                self._buffer.clear()
                self._fatal_error(exc, 'Fatal write error on SSL transport')
                return
            if n:
                del self._buffer[:n]
        self._maybe_resume_protocol()
        if not self._buffer:
            self._loop.remove_writer(self._sock_fd)
            if self._closing:
                self._call_connection_lost(None)

    def write(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError('data argument must be byte-ish (%r)', type(data))
        if not data:
            return
        if self._conn_lost:
            if self._conn_lost >= constants.LOG_THRESHOLD_FOR_CONNLOST_WRITES:
                logger.warning('socket.send() raised exception.')
            self._conn_lost += 1
            return
        if not self._buffer:
            self._loop.add_writer(self._sock_fd, self._write_ready)
        self._buffer.extend(data)
        self._maybe_pause_protocol()

    def can_write_eof(self):
        return False