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
class AsyncIOLoop(BaseAsyncIOLoop):
    """``AsyncIOLoop`` is an `.IOLoop` that runs on an ``asyncio`` event loop.
    This class follows the usual Tornado semantics for creating new
    ``IOLoops``; these loops are not necessarily related to the
    ``asyncio`` default event loop.

    Each ``AsyncIOLoop`` creates a new ``asyncio.EventLoop``; this object
    can be accessed with the ``asyncio_loop`` attribute.

    .. versionchanged:: 6.2

       Support explicit ``asyncio_loop`` argument
       for specifying the asyncio loop to attach to,
       rather than always creating a new one with the default policy.

    .. versionchanged:: 5.0

       When an ``AsyncIOLoop`` becomes the current `.IOLoop`, it also sets
       the current `asyncio` event loop.

    .. deprecated:: 5.0

       Now used automatically when appropriate; it is no longer necessary
       to refer to this class directly.
    """

    def initialize(self, **kwargs: Any) -> None:
        self.is_current = False
        loop = None
        if 'asyncio_loop' not in kwargs:
            kwargs['asyncio_loop'] = loop = asyncio.new_event_loop()
        try:
            super().initialize(**kwargs)
        except Exception:
            if loop is not None:
                loop.close()
            raise

    def close(self, all_fds: bool=False) -> None:
        if self.is_current:
            self._clear_current()
        super().close(all_fds=all_fds)

    def _make_current(self) -> None:
        if not self.is_current:
            try:
                self.old_asyncio = asyncio.get_event_loop()
            except (RuntimeError, AssertionError):
                self.old_asyncio = None
            self.is_current = True
        asyncio.set_event_loop(self.asyncio_loop)

    def _clear_current_hook(self) -> None:
        if self.is_current:
            asyncio.set_event_loop(self.old_asyncio)
            self.is_current = False