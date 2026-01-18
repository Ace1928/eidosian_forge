import asyncio
import atexit
import concurrent.futures
import errno
import functools
import select
import socket
import sys
import threading
import typing
import warnings
from tornado.gen import convert_yielded
from tornado.ioloop import IOLoop, _Selectable
from typing import (
class AsyncIOMainLoop(BaseAsyncIOLoop):
    """``AsyncIOMainLoop`` creates an `.IOLoop` that corresponds to the
    current ``asyncio`` event loop (i.e. the one returned by
    ``asyncio.get_event_loop()``).

    .. deprecated:: 5.0

       Now used automatically when appropriate; it is no longer necessary
       to refer to this class directly.

    .. versionchanged:: 5.0

       Closing an `AsyncIOMainLoop` now closes the underlying asyncio loop.
    """

    def initialize(self, **kwargs: Any) -> None:
        super().initialize(asyncio.get_event_loop(), **kwargs)

    def _make_current(self) -> None:
        pass