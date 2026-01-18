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
@contextmanager
def _join_started_threads() -> Iterator[None]:
    before = frozenset(threading.enumerate())
    try:
        yield
    finally:
        for thread in threading.enumerate():
            if thread not in before:
                thread.join(timeout=1.0)
                assert not thread.is_alive()