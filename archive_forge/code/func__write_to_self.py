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
def _write_to_self(self):
    csock = self._csock
    if csock is None:
        return
    try:
        csock.send(b'\x00')
    except OSError:
        if self._debug:
            logger.debug('Fail to write a null byte into the self-pipe socket', exc_info=True)