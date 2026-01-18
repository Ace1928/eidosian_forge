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
def _close_self_pipe(self):
    if self._self_reading_future is not None:
        self._self_reading_future.cancel()
        self._self_reading_future = None
    self._ssock.close()
    self._ssock = None
    self._csock.close()
    self._csock = None
    self._internal_fds -= 1