from asyncio import AbstractEventLoop, new_event_loop, run_coroutine_threadsafe
from concurrent.futures import Future
from threading import Thread, Lock
from typing import ContextManager, Generic, TypeVar, Optional, Callable
class _ManagedEventLoopImpl(ContextManager):
    _loop: AbstractEventLoop
    _thread: Thread

    def __init__(self, name=None):
        self._loop = new_event_loop()
        self._thread = Thread(target=lambda: self._loop.run_forever(), name=name, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def submit(self, coro) -> Future:
        return run_coroutine_threadsafe(coro, self._loop)