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
class _ProactorBaseWritePipeTransport(_ProactorBasePipeTransport, transports.WriteTransport):
    """Transport for write pipes."""
    _start_tls_compatible = True

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._empty_waiter = None

    def write(self, data):
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f'data argument must be a bytes-like object, not {type(data).__name__}')
        if self._eof_written:
            raise RuntimeError('write_eof() already called')
        if self._empty_waiter is not None:
            raise RuntimeError('unable to write; sendfile is in progress')
        if not data:
            return
        if self._conn_lost:
            if self._conn_lost >= constants.LOG_THRESHOLD_FOR_CONNLOST_WRITES:
                logger.warning('socket.send() raised exception.')
            self._conn_lost += 1
            return
        if self._write_fut is None:
            assert self._buffer is None
            self._loop_writing(data=bytes(data))
        elif not self._buffer:
            self._buffer = bytearray(data)
            self._maybe_pause_protocol()
        else:
            self._buffer.extend(data)
            self._maybe_pause_protocol()

    def _loop_writing(self, f=None, data=None):
        try:
            if f is not None and self._write_fut is None and self._closing:
                return
            assert f is self._write_fut
            self._write_fut = None
            self._pending_write = 0
            if f:
                f.result()
            if data is None:
                data = self._buffer
                self._buffer = None
            if not data:
                if self._closing:
                    self._loop.call_soon(self._call_connection_lost, None)
                if self._eof_written:
                    self._sock.shutdown(socket.SHUT_WR)
                self._maybe_resume_protocol()
            else:
                self._write_fut = self._loop._proactor.send(self._sock, data)
                if not self._write_fut.done():
                    assert self._pending_write == 0
                    self._pending_write = len(data)
                    self._write_fut.add_done_callback(self._loop_writing)
                    self._maybe_pause_protocol()
                else:
                    self._write_fut.add_done_callback(self._loop_writing)
            if self._empty_waiter is not None and self._write_fut is None:
                self._empty_waiter.set_result(None)
        except ConnectionResetError as exc:
            self._force_close(exc)
        except OSError as exc:
            self._fatal_error(exc, 'Fatal write error on pipe transport')

    def can_write_eof(self):
        return True

    def write_eof(self):
        self.close()

    def abort(self):
        self._force_close(None)

    def _make_empty_waiter(self):
        if self._empty_waiter is not None:
            raise RuntimeError('Empty waiter is already set')
        self._empty_waiter = self._loop.create_future()
        if self._write_fut is None:
            self._empty_waiter.set_result(None)
        return self._empty_waiter

    def _reset_empty_waiter(self):
        self._empty_waiter = None