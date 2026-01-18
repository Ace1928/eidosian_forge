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
def _add_recv_event(self, kind, kwargs=None, future=None):
    """Add a recv event, returning the corresponding Future"""
    f = future or self._Future()
    if kind.startswith('recv') and kwargs.get('flags', 0) & _zmq.DONTWAIT:
        recv = getattr(self._shadow_sock, kind)
        try:
            r = recv(**kwargs)
        except Exception as e:
            f.set_exception(e)
        else:
            f.set_result(r)
        return f
    timer = _NoTimer
    if hasattr(_zmq, 'RCVTIMEO'):
        timeout_ms = self._shadow_sock.rcvtimeo
        if timeout_ms >= 0:
            timer = self._add_timeout(f, timeout_ms * 0.001)
    _future_event = _FutureEvent(f, kind, kwargs, msg=None, timer=timer)
    self._recv_futures.append(_future_event)
    if self._shadow_sock.get(EVENTS) & POLLIN:
        self._handle_recv()
    if self._recv_futures and _future_event in self._recv_futures:
        f.add_done_callback(partial(self._remove_finished_future, event_list=self._recv_futures, event=_future_event))
        self._add_io_state(POLLIN)
    return f