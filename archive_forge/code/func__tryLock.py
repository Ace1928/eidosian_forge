from __future__ import annotations
import inspect
import traceback
import warnings
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop, Future, iscoroutine
from contextvars import Context as _Context, copy_context as _copy_context
from enum import Enum
from functools import wraps
from sys import exc_info, implementation
from types import CoroutineType, GeneratorType, MappingProxyType, TracebackType
from typing import (
import attr
from incremental import Version
from typing_extensions import Concatenate, Literal, ParamSpec, Self
from twisted.internet.interfaces import IDelayedCall, IReactorTime
from twisted.logger import Logger
from twisted.python import lockfile
from twisted.python.compat import _PYPY, cmp, comparable
from twisted.python.deprecate import deprecated, warnAboutFunction
from twisted.python.failure import Failure, _extraneous
def _tryLock() -> None:
    if self.lock():
        if self._timeoutCall is not None:
            self._timeoutCall.cancel()
            self._timeoutCall = None
        self._tryLockCall = None
        d.callback(None)
    else:
        if timeout is not None and self._timeoutCall is None:
            reason = Failure(TimeoutError('Timed out acquiring lock: %s after %fs' % (self.name, timeout)))
            self._timeoutCall = self._scheduler.callLater(timeout, _cancelLock, reason)
        self._tryLockCall = self._scheduler.callLater(self._interval, _tryLock)