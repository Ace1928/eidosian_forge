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
def _startRunCallbacks(self, result: object) -> None:
    if self.called:
        if self._suppressAlreadyCalled:
            self._suppressAlreadyCalled = False
            return
        if self.debug:
            if self._debugInfo is None:
                self._debugInfo = DebugInfo()
            extra = '\n' + self._debugInfo._getDebugTracebacks()
            raise AlreadyCalledError(extra)
        raise AlreadyCalledError
    if self.debug:
        if self._debugInfo is None:
            self._debugInfo = DebugInfo()
        self._debugInfo.invoker = traceback.format_stack()[:-2]
    self.called = True
    self._canceller = None
    self.result = result
    self._runCallbacks()