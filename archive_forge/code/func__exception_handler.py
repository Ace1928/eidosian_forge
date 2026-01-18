from __future__ import annotations
import asyncio
import functools
import logging
import sys
import typing
from .abstract_loop import EventLoop, ExitMainLoop
def _exception_handler(self, loop: asyncio.AbstractEventLoop, context):
    exc = context.get('exception')
    if exc:
        loop.stop()
        if self._idle_asyncio_handle:
            self._idle_asyncio_handle.cancel()
            self._idle_asyncio_handle = None
        if not isinstance(exc, ExitMainLoop):
            self._exc = exc
    else:
        loop.default_exception_handler(context)