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
def _read_ready__on_eof(self):
    if self._loop.get_debug():
        logger.debug('%r received EOF', self)
    try:
        keep_open = self._protocol.eof_received()
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        self._fatal_error(exc, 'Fatal error: protocol.eof_received() call failed.')
        return
    if keep_open:
        self._loop._remove_reader(self._sock_fd)
    else:
        self.close()