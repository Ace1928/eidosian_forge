from __future__ import annotations
import asyncio
import warnings
from typing import Any
from tornado.concurrent import Future
from tornado.ioloop import IOLoop
import zmq as _zmq
from zmq._future import _AsyncPoller, _AsyncSocket
def _watch_raw_socket(self, loop, socket, evt, f):
    """Schedule callback for a raw socket"""
    loop.add_handler(socket, lambda *args: f(), evt)