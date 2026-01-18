from __future__ import annotations
import collections.abc
import inspect
import os
import signal
import threading
from abc import ABCMeta
from functools import update_wrapper
from typing import (
from sniffio import thread_local as sniffio_loop
import trio
class ConflictDetector:
    """Detect when two tasks are about to perform operations that would
    conflict.

    Use as a synchronous context manager; if two tasks enter it at the same
    time then the second one raises an error. You can use it when there are
    two pieces of code that *would* collide and need a lock if they ever were
    called at the same time, but that should never happen.

    We use this in particular for things like, making sure that two different
    tasks don't call sendall simultaneously on the same stream.

    """

    def __init__(self, msg: str) -> None:
        self._msg = msg
        self._held = False

    def __enter__(self) -> None:
        if self._held:
            raise trio.BusyResourceError(self._msg)
        else:
            self._held = True

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self._held = False