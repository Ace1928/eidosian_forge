from __future__ import annotations
import asyncio
import warnings
from typing import Any
from tornado.concurrent import Future
from tornado.ioloop import IOLoop
import zmq as _zmq
from zmq._future import _AsyncPoller, _AsyncSocket
@staticmethod
def _socket_class(self, socket_type):
    return Socket(self, socket_type)