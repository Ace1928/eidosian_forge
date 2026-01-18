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
def _stop_accept_futures(self):
    for future in self._accept_futures.values():
        future.cancel()
    self._accept_futures.clear()