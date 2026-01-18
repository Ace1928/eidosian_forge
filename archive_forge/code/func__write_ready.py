import collections
import errno
import functools
import selectors
import socket
import warnings
import weakref
from . import base_events
from . import constants
from . import events
from . import futures
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
def _write_ready(self):
    assert self._buffer, 'Data should not be empty'
    if self._conn_lost:
        return
    try:
        n = self._sock.send(self._buffer)
    except (BlockingIOError, InterruptedError):
        pass
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        self._loop._remove_writer(self._sock_fd)
        self._buffer.clear()
        self._fatal_error(exc, 'Fatal write error on socket transport')
        if self._empty_waiter is not None:
            self._empty_waiter.set_exception(exc)
    else:
        if n:
            del self._buffer[:n]
        self._maybe_resume_protocol()
        if not self._buffer:
            self._loop._remove_writer(self._sock_fd)
            if self._empty_waiter is not None:
                self._empty_waiter.set_result(None)
            if self._closing:
                self._call_connection_lost(None)
            elif self._eof:
                self._sock.shutdown(socket.SHUT_WR)