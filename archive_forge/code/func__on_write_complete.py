import asyncio
import logging
import re
import types
from tornado.concurrent import (
from tornado.escape import native_str, utf8
from tornado import gen
from tornado import httputil
from tornado import iostream
from tornado.log import gen_log, app_log
from tornado.util import GzipDecompressor
from typing import cast, Optional, Type, Awaitable, Callable, Union, Tuple
def _on_write_complete(self, future: 'Future[None]') -> None:
    exc = future.exception()
    if exc is not None and (not isinstance(exc, iostream.StreamClosedError)):
        future.result()
    if self._write_callback is not None:
        callback = self._write_callback
        self._write_callback = None
        self.stream.io_loop.add_callback(callback)
    if self._write_future is not None:
        future = self._write_future
        self._write_future = None
        future_set_result_unless_cancelled(future, None)