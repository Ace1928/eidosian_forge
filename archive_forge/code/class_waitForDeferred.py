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
class waitForDeferred:
    """
    See L{deferredGenerator}.
    """
    result: Any = _NO_RESULT

    def __init__(self, d: Deferred[object]) -> None:
        warnings.warn('twisted.internet.defer.waitForDeferred was deprecated in Twisted 15.0.0; please use twisted.internet.defer.inlineCallbacks instead', DeprecationWarning, stacklevel=2)
        if not isinstance(d, Deferred):
            raise TypeError(f'You must give waitForDeferred a Deferred. You gave it {d!r}.')
        self.d = d

    def getResult(self) -> Any:
        if isinstance(self.result, Failure):
            self.result.raiseException()
        self.result is not _NO_RESULT
        return self.result