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
def _schedule_remaining_events(self, events=None):
    """Schedule a call to handle_events next loop iteration

        If there are still events to handle.
        """
    if self._state == 0:
        return
    if events is None:
        events = self._shadow_sock.get(EVENTS)
    if events & self._state:
        self._call_later(0, self._handle_events)