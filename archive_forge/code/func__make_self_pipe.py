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
def _make_self_pipe(self):
    self._ssock, self._csock = socket.socketpair()
    self._ssock.setblocking(False)
    self._csock.setblocking(False)
    self._internal_fds += 1