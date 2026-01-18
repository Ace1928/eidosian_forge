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
def _stop_serving(self, sock):
    future = self._accept_futures.pop(sock.fileno(), None)
    if future:
        future.cancel()
    self._proactor._stop_serving(sock)
    sock.close()