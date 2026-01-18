from __future__ import annotations
import asyncio
import selectors
import sys
import warnings
from asyncio import Future, SelectorEventLoop
from weakref import WeakKeyDictionary
import zmq as _zmq
from zmq import _future
def _close_selector_and_loop():
    asyncio_loop.close = loop_close
    _selectors.pop(asyncio_loop, None)
    selector_loop.close()