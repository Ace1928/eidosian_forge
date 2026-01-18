from __future__ import annotations
import asyncio
import selectors
import sys
import warnings
from asyncio import Future, SelectorEventLoop
from weakref import WeakKeyDictionary
import zmq as _zmq
from zmq import _future
def _get_selector(self, io_loop=None):
    if io_loop is None:
        io_loop = self._get_loop()
    return _get_selector(io_loop)