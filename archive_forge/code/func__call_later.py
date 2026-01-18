from __future__ import annotations
import asyncio
import warnings
from typing import Any
from tornado.concurrent import Future
from tornado.ioloop import IOLoop
import zmq as _zmq
from zmq._future import _AsyncPoller, _AsyncSocket
def _call_later(self, delay, callback):
    io_loop = self._get_loop()
    timeout = io_loop.call_later(delay, callback)
    return _CancellableTornadoTimeout(io_loop, timeout)