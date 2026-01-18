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