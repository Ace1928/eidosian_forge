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
def _loop_reading(self, fut=None):
    data = None
    try:
        if self._conn_lost:
            return
        assert self._read_fut is fut or (self._read_fut is None and self._closing)
        self._read_fut = None
        if fut is not None:
            res = fut.result()
            if self._closing:
                data = None
                return
            if self._address is not None:
                data, addr = (res, self._address)
            else:
                data, addr = res
        if self._conn_lost:
            return
        if self._address is not None:
            self._read_fut = self._loop._proactor.recv(self._sock, self.max_size)
        else:
            self._read_fut = self._loop._proactor.recvfrom(self._sock, self.max_size)
    except OSError as exc:
        self._protocol.error_received(exc)
    except exceptions.CancelledError:
        if not self._closing:
            raise
    else:
        if self._read_fut is not None:
            self._read_fut.add_done_callback(self._loop_reading)
    finally:
        if data:
            self._protocol.datagram_received(data, addr)