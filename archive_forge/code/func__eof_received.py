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
def _eof_received(self):
    if self._loop.get_debug():
        logger.debug('%r received EOF', self)
    try:
        keep_open = self._protocol.eof_received()
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        self._fatal_error(exc, 'Fatal error: protocol.eof_received() call failed.')
        return
    if not keep_open:
        self.close()