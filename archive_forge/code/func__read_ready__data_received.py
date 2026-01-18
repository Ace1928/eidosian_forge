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
def _read_ready__data_received(self):
    if self._conn_lost:
        return
    try:
        data = self._sock.recv(self.max_size)
    except (BlockingIOError, InterruptedError):
        return
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        self._fatal_error(exc, 'Fatal read error on socket transport')
        return
    if not data:
        self._read_ready__on_eof()
        return
    try:
        self._protocol.data_received(data)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        self._fatal_error(exc, 'Fatal error: protocol.data_received() call failed.')