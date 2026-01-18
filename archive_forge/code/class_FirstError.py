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
@comparable
class FirstError(Exception):
    """
    First error to occur in a L{DeferredList} if C{fireOnOneErrback} is set.

    @ivar subFailure: The L{Failure} that occurred.
    @ivar index: The index of the L{Deferred} in the L{DeferredList} where
        it happened.
    """

    def __init__(self, failure: Failure, index: int) -> None:
        Exception.__init__(self, failure, index)
        self.subFailure = failure
        self.index = index

    def __repr__(self) -> str:
        """
        The I{repr} of L{FirstError} instances includes the repr of the
        wrapped failure's exception and the index of the L{FirstError}.
        """
        return 'FirstError[#%d, %r]' % (self.index, self.subFailure.value)

    def __str__(self) -> str:
        """
        The I{str} of L{FirstError} instances includes the I{str} of the
        entire wrapped failure (including its traceback and exception) and
        the index of the L{FirstError}.
        """
        return 'FirstError[#%d, %s]' % (self.index, self.subFailure)

    def __cmp__(self, other: object) -> int:
        """
        Comparison between L{FirstError} and other L{FirstError} instances
        is defined as the comparison of the index and sub-failure of each
        instance.  L{FirstError} instances don't compare equal to anything
        that isn't a L{FirstError} instance.

        @since: 8.2
        """
        if isinstance(other, FirstError):
            return cmp((self.index, self.subFailure), (other.index, other.subFailure))
        return -1