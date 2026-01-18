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
def _parseDeferredListResult(resultList: List[_DeferredListResultItemT[_T]], fireOnOneErrback: bool=False, /) -> List[_T]:
    if __debug__:
        for result in resultList:
            assert result is not None
            success, value = result
            assert success
    return [x[1] for x in resultList]