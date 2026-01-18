import asyncio
import asyncio.coroutines
import contextvars
import functools
import inspect
import os
import sys
import threading
import warnings
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
from .current_thread_executor import CurrentThreadExecutor
from .local import Local
def _run_event_loop(self, loop, coro):
    """
        Runs the given event loop (designed to be called in a thread).
        """
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(coro)
    finally:
        try:
            tasks = asyncio.all_tasks(loop)
            for task in tasks:
                task.cancel()

            async def gather():
                await asyncio.gather(*tasks, return_exceptions=True)
            loop.run_until_complete(gather())
            for task in tasks:
                if task.cancelled():
                    continue
                if task.exception() is not None:
                    loop.call_exception_handler({'message': 'unhandled exception during loop shutdown', 'exception': task.exception(), 'task': task})
            if hasattr(loop, 'shutdown_asyncgens'):
                loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
            asyncio.set_event_loop(self.main_event_loop)