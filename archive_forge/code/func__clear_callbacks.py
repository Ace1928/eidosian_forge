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
def _clear_callbacks(self) -> None:
    """Clears the callback attributes.

        This allows the request handler to be garbage collected more
        quickly in CPython by breaking up reference cycles.
        """
    self._write_callback = None
    self._write_future = None
    self._close_callback = None
    if self.stream is not None:
        self.stream.set_close_callback(None)