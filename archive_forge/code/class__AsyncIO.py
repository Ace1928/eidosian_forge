from __future__ import annotations
import asyncio
import selectors
import sys
import warnings
from asyncio import Future, SelectorEventLoop
from weakref import WeakKeyDictionary
import zmq as _zmq
from zmq import _future
class _AsyncIO:
    _Future = Future
    _WRITE = selectors.EVENT_WRITE
    _READ = selectors.EVENT_READ

    def _default_loop(self):
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            warnings.warn('No running event\xa0loop. zmq.asyncio should be used from within an asyncio loop.', RuntimeWarning, stacklevel=4)
        return asyncio.get_event_loop()