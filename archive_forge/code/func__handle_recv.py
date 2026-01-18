from __future__ import annotations
import asyncio
import pickle
import warnings
from queue import Queue
from typing import Any, Awaitable, Callable, Sequence, cast, overload
from tornado.ioloop import IOLoop
from tornado.log import gen_log
import zmq
import zmq._future
from zmq import POLLIN, POLLOUT
from zmq._typing import Literal
from zmq.utils import jsonapi
def _handle_recv(self):
    """Handle a recv event."""
    if self._flushed:
        return
    try:
        msg = self.socket.recv_multipart(zmq.NOBLOCK, copy=self._recv_copy)
    except zmq.ZMQError as e:
        if e.errno == zmq.EAGAIN:
            pass
        else:
            raise
    else:
        if self._recv_callback:
            callback = self._recv_callback
            self._run_callback(callback, msg)