from __future__ import annotations
import asyncio
import warnings
from typing import Any
from tornado.concurrent import Future
from tornado.ioloop import IOLoop
import zmq as _zmq
from zmq._future import _AsyncPoller, _AsyncSocket
class _TornadoFuture(Future):
    """Subclass Tornado Future, reinstating cancellation."""

    def cancel(self):
        if self.done():
            return False
        self.set_exception(CancelledError())
        return True

    def cancelled(self):
        return self.done() and isinstance(self.exception(), CancelledError)