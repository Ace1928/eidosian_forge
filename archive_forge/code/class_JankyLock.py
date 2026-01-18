from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from queue import Queue
from typing import TYPE_CHECKING, Iterator, NoReturn
import pytest
from .. import _thread_cache
from .._thread_cache import ThreadCache, start_thread_soon
from .tutil import gc_collect_harder, slow
class JankyLock:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter = 3

    def acquire(self, timeout: int=-1) -> bool:
        got_it = self._lock.acquire(timeout=timeout)
        if timeout == -1:
            return True
        elif got_it:
            if self._counter > 0:
                self._counter -= 1
                self._lock.release()
                return False
            return True
        else:
            return False

    def release(self) -> None:
        self._lock.release()