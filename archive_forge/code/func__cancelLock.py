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
def _cancelLock(reason: Union[Failure, Exception]) -> None:
    """
            Cancel a L{DeferredFilesystemLock.deferUntilLocked} call.

            @type reason: L{Failure}
            @param reason: The reason why the call is cancelled.
            """
    assert self._tryLockCall is not None
    self._tryLockCall.cancel()
    self._tryLockCall = None
    if self._timeoutCall is not None and self._timeoutCall.active():
        self._timeoutCall.cancel()
        self._timeoutCall = None
    if self.lock():
        d.callback(None)
    else:
        d.errback(reason)