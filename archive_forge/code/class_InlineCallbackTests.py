from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
class InlineCallbackTests(unittest.SynchronousTestCase):

    def test_inlineCallbacksTracebacks(self) -> None:
        """
        L{defer.inlineCallbacks} that re-raise tracebacks into their deferred
        should not lose their tracebacks.
        """
        f = getDivisionFailure()
        d: Deferred[None] = Deferred()
        try:
            f.raiseException()
        except BaseException:
            d.errback()

        def ic(d: object) -> Generator[Any, Any, None]:
            yield d
        defer.inlineCallbacks(ic)
        newFailure = self.failureResultOf(d)
        tb = traceback.extract_tb(newFailure.getTracebackObject())
        self.assertEqual(len(tb), 3)
        self.assertIn('test_defer', tb[2][0])
        self.assertEqual('getDivisionFailure', tb[2][2])
        self.assertEqual('1 / 0', tb[2][3])
        self.assertIn('test_defer', tb[0][0])
        self.assertEqual('test_inlineCallbacksTracebacks', tb[0][2])
        self.assertEqual('f.raiseException()', tb[0][3])

    def test_fromCoroutineRequiresCoroutine(self) -> None:
        """
        L{Deferred.fromCoroutine} requires a coroutine object or a generator,
        and will reject things that are not that.
        """
        thingsThatAreNotCoroutines = [lambda x: x, 1, True, self.test_fromCoroutineRequiresCoroutine, None, defer]
        for thing in thingsThatAreNotCoroutines:
            self.assertRaises(defer.NotACoroutineError, Deferred.fromCoroutine, thing)

    def test_inlineCallbacksCancelCaptured(self) -> None:
        """
        Cancelling an L{defer.inlineCallbacks} correctly handles the function
        catching the L{defer.CancelledError}.

        The desired behavior is:
            1. If the function is waiting on an inner deferred, that inner
               deferred is cancelled, and a L{defer.CancelledError} is raised
               within the function.
            2. If the function catches that exception, execution continues, and
               the deferred returned by the function is not resolved.
            3. Cancelling the deferred again cancels any deferred the function
               is waiting on, and the exception is raised.
        """
        canceller1Calls: List[Deferred[object]] = []
        canceller2Calls: List[Deferred[object]] = []
        d1: Deferred[object] = Deferred(canceller1Calls.append)
        d2: Deferred[object] = Deferred(canceller2Calls.append)

        @defer.inlineCallbacks
        def testFunc() -> Generator[Deferred[object], object, None]:
            try:
                yield d1
            except Exception:
                pass
            yield d2
        funcD = testFunc()
        self.assertNoResult(d1)
        self.assertNoResult(d2)
        self.assertNoResult(funcD)
        self.assertEqual(canceller1Calls, [])
        self.assertEqual(canceller1Calls, [])
        funcD.cancel()
        self.assertEqual(canceller1Calls, [d1])
        self.assertEqual(canceller2Calls, [])
        self.assertNoResult(funcD)
        funcD.cancel()
        failure = self.failureResultOf(funcD)
        self.assertEqual(failure.type, defer.CancelledError)
        self.assertEqual(canceller2Calls, [d2])

    @pyunit.skipIf(_PYPY, 'GC works differently on PyPy.')
    def test_inlineCallbacksNoCircularReference(self) -> None:
        """
        When using L{defer.inlineCallbacks}, after the function exits, it will
        not keep references to the function itself or the arguments.

        This ensures that the machinery gets deallocated immediately rather than
        waiting for a GC, on CPython.

        The GC on PyPy works differently (del doesn't immediately deallocate the
        object), so we skip the test.
        """
        obj: Set[Any] = set()
        objWeakRef = weakref.ref(obj)

        @defer.inlineCallbacks
        def func(a: Any) -> Any:
            yield a
            return a
        funcD = func(obj)
        self.assertEqual(obj, self.successResultOf(funcD))
        funcDWeakRef = weakref.ref(funcD)
        del obj
        del funcD
        self.assertIsNone(objWeakRef())
        self.assertIsNone(funcDWeakRef())

    @pyunit.skipIf(_PYPY, 'GC works differently on PyPy.')
    def test_coroutineNoCircularReference(self) -> None:
        """
        Tests that there is no circular dependency when using
        L{Deferred.fromCoroutine}, so that the machinery gets cleaned up
        immediately rather than waiting for a GC.
        """
        obj: Set[Any] = set()
        objWeakRef = weakref.ref(obj)

        async def func(a: Any) -> Any:
            return a
        funcD = Deferred.fromCoroutine(func(obj))
        self.assertEqual(obj, self.successResultOf(funcD))
        funcDWeakRef = weakref.ref(funcD)
        del obj
        del funcD
        self.assertIsNone(objWeakRef())
        self.assertIsNone(funcDWeakRef())