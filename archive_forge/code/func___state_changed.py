from __future__ import annotations
import sys
import time
import warnings
import gevent
from gevent.event import AsyncResult
from gevent.hub import get_hub
import zmq
from zmq import Context as _original_Context
from zmq import Socket as _original_Socket
from .poll import _Poller
def __state_changed(self, event=None, _evtype=None):
    if self.closed:
        self.__cleanup_events()
        return
    try:
        events = super().getsockopt(zmq.EVENTS)
    except zmq.ZMQError as exc:
        self.__writable.set_exception(exc)
        self.__readable.set_exception(exc)
    else:
        if events & zmq.POLLOUT:
            self.__writable.set()
        if events & zmq.POLLIN:
            self.__readable.set()