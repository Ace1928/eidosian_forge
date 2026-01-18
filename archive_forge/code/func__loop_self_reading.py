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
def _loop_self_reading(self, f=None):
    try:
        if f is not None:
            f.result()
        if self._self_reading_future is not f:
            return
        f = self._proactor.recv(self._ssock, 4096)
    except exceptions.CancelledError:
        return
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        self.call_exception_handler({'message': 'Error on reading from the event loop self pipe', 'exception': exc, 'loop': self})
    else:
        self._self_reading_future = f
        f.add_done_callback(self._loop_self_reading)