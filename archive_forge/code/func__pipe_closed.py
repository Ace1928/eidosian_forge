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
def _pipe_closed(self, fut):
    if fut.cancelled():
        return
    assert fut.result() == b''
    if self._closing:
        assert self._read_fut is None
        return
    assert fut is self._read_fut, (fut, self._read_fut)
    self._read_fut = None
    if self._write_fut is not None:
        self._force_close(BrokenPipeError())
    else:
        self.close()