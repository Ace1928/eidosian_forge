import abc
import asyncio
import base64
import hashlib
import os
import sys
import struct
import tornado
from urllib.parse import urlparse
import warnings
import zlib
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.escape import utf8, native_str, to_unicode
from tornado import gen, httpclient, httputil
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.iostream import StreamClosedError, IOStream
from tornado.log import gen_log, app_log
from tornado.netutil import Resolver
from tornado import simple_httpclient
from tornado.queues import Queue
from tornado.tcpclient import TCPClient
from tornado.util import _websocket_mask
from typing import (
from types import TracebackType
class _WebSocketDelegate(Protocol):

    def on_ws_connection_close(self, close_code: Optional[int]=None, close_reason: Optional[str]=None) -> None:
        pass

    def on_message(self, message: Union[str, bytes]) -> Optional['Awaitable[None]']:
        pass

    def on_ping(self, data: bytes) -> None:
        pass

    def on_pong(self, data: bytes) -> None:
        pass

    def log_exception(self, typ: Optional[Type[BaseException]], value: Optional[BaseException], tb: Optional[TracebackType]) -> None:
        pass