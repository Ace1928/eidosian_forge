import asyncio as __asyncio
import typing as _typing
import sys as _sys
import warnings as _warnings
from asyncio.events import BaseDefaultEventLoopPolicy as __BasePolicy
from . import includes as __includes  # NOQA
from .loop import Loop as __BaseLoop  # NOQA
from ._version import __version__  # NOQA
def _cancel_all_tasks(loop: __asyncio.AbstractEventLoop) -> None:
    to_cancel = __asyncio.all_tasks(loop)
    if not to_cancel:
        return
    for task in to_cancel:
        task.cancel()
    loop.run_until_complete(__asyncio.gather(*to_cancel, return_exceptions=True))
    for task in to_cancel:
        if task.cancelled():
            continue
        if task.exception() is not None:
            loop.call_exception_handler({'message': 'unhandled exception during asyncio.run() shutdown', 'exception': task.exception(), 'task': task})