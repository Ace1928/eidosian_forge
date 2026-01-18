from __future__ import annotations
import warnings
from asyncio import Future
from collections import deque
from functools import partial
from itertools import chain
from typing import Any, Awaitable, Callable, NamedTuple, TypeVar, cast, overload
import zmq as _zmq
from zmq import EVENTS, POLLIN, POLLOUT
from zmq._typing import Literal
def _clear_io_state(self):
    """unregister the ioloop event handler

        called once during close
        """
    fd = self._shadow_sock
    if self._shadow_sock.closed:
        fd = self._fd
    if self._current_loop is not None:
        self._current_loop.remove_handler(fd)