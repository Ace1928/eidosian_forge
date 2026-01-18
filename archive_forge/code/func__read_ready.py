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
def _read_ready(self):
    if self._conn_lost:
        return
    try:
        data, addr = self._sock.recvfrom(self.max_size)
    except (BlockingIOError, InterruptedError):
        pass
    except OSError as exc:
        self._protocol.error_received(exc)
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        self._fatal_error(exc, 'Fatal read error on datagram transport')
    else:
        self._protocol.datagram_received(data, addr)