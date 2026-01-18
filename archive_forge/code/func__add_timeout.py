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
def _add_timeout(self, future, timeout):
    """Add a timeout for a send or recv Future"""

    def future_timeout():
        if future.done():
            return
        future.set_exception(_zmq.Again())
    return self._call_later(timeout, future_timeout)