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
def _add_io_state(self, state):
    """Add io_state to poller."""
    self._state = self._state | state
    self._update_handler(self._state)