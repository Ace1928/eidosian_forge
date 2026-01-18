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
def _cbDeferred(self, result: _SelfResultT, index: int, succeeded: bool) -> Optional[_SelfResultT]:
    """
        (internal) Callback for when one of my deferreds fires.
        """
    self.resultList[index] = (succeeded, result)
    self.finishedCount += 1
    if not self.called:
        if succeeded == SUCCESS and self.fireOnOneCallback:
            self.callback((result, index))
        elif succeeded == FAILURE and self.fireOnOneErrback:
            assert isinstance(result, Failure)
            self.errback(Failure(FirstError(result, index)))
        elif self.finishedCount == len(self.resultList):
            self.callback(cast(_DeferredListResultListT[Any], self.resultList))
    if succeeded == FAILURE and self.consumeErrors:
        return None
    return result