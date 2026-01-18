from __future__ import annotations
import contextlib
import logging
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import local
from typing import TYPE_CHECKING, Any, ClassVar
from weakref import WeakValueDictionary
from ._error import Timeout
class AcquireReturnProxy:
    """A context aware object that will release the lock file when exiting."""

    def __init__(self, lock: BaseFileLock) -> None:
        self.lock = lock

    def __enter__(self) -> BaseFileLock:
        return self.lock

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        self.lock.release()