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
class DeferredFutureAdapterTests(unittest.TestCase):

    def newLoop(self) -> AbstractEventLoop:
        """
        Create a new event loop that will be closed at the end of the test.
        """
        result = _new_event_loop()
        self.addCleanup(result.close)
        return result

    def test_asFuture(self) -> None:
        """
        L{Deferred.asFuture} returns a L{asyncio.Future} which fires when
        the given L{Deferred} does.
        """
        d: Deferred[int] = Deferred()
        loop = self.newLoop()
        aFuture = d.asFuture(loop)
        self.assertEqual(aFuture.done(), False)
        d.callback(13)
        callAllSoonCalls(loop)
        self.assertEqual(self.successResultOf(d), None)
        self.assertEqual(aFuture.result(), 13)

    def test_asFutureCancelFuture(self) -> None:
        """
        L{Deferred.asFuture} returns a L{asyncio.Future} which, when
        cancelled, will cancel the original L{Deferred}.
        """
        called = False

        def canceler(dprime: Deferred[object]) -> None:
            nonlocal called
            called = True
        d: Deferred[None] = Deferred(canceler)
        loop = self.newLoop()
        aFuture = d.asFuture(loop)
        aFuture.cancel()
        callAllSoonCalls(loop)
        self.assertTrue(called)
        self.assertEqual(self.successResultOf(d), None)
        self.assertRaises(CancelledError, aFuture.result)

    def test_asFutureSuccessCancel(self) -> None:
        """
        While Futures don't support succeeding in response to cancellation,
        Deferreds do; if a Deferred is coerced into a success by a Future
        cancellation, that should just be ignored.
        """

        def canceler(dprime: Deferred[object]) -> None:
            dprime.callback(9)
        d: Deferred[None] = Deferred(canceler)
        loop = self.newLoop()
        aFuture = d.asFuture(loop)
        aFuture.cancel()
        callAllSoonCalls(loop)
        self.assertEqual(self.successResultOf(d), None)
        self.assertRaises(CancelledError, aFuture.result)

    def test_asFutureFailure(self) -> None:
        """
        L{Deferred.asFuture} makes a L{asyncio.Future} fire with an
        exception when the given L{Deferred} does.
        """
        d: Deferred[None] = Deferred()
        theFailure = Failure(ZeroDivisionError())
        loop = self.newLoop()
        future = d.asFuture(loop)
        callAllSoonCalls(loop)
        d.errback(theFailure)
        callAllSoonCalls(loop)
        self.assertRaises(ZeroDivisionError, future.result)

    def test_fromFuture(self) -> None:
        """
        L{Deferred.fromFuture} returns a L{Deferred} that fires
        when the given L{asyncio.Future} does.
        """
        loop = self.newLoop()
        aFuture: Future[int] = Future(loop=loop)
        d = Deferred.fromFuture(aFuture)
        self.assertNoResult(d)
        aFuture.set_result(7)
        callAllSoonCalls(loop)
        self.assertEqual(self.successResultOf(d), 7)

    def test_fromFutureFutureCancelled(self) -> None:
        """
        L{Deferred.fromFuture} makes a L{Deferred} fire with
        an L{asyncio.CancelledError} when the given
        L{asyncio.Future} is cancelled.
        """
        loop = self.newLoop()
        cancelled: Future[None] = Future(loop=loop)
        d = Deferred.fromFuture(cancelled)
        cancelled.cancel()
        callAllSoonCalls(loop)
        self.assertRaises(CancelledError, cancelled.result)
        self.failureResultOf(d).trap(CancelledError)

    def test_fromFutureDeferredCancelled(self) -> None:
        """
        L{Deferred.fromFuture} makes a L{Deferred} which, when
        cancelled, cancels the L{asyncio.Future} it was created from.
        """
        loop = self.newLoop()
        cancelled: Future[None] = Future(loop=loop)
        d = Deferred.fromFuture(cancelled)
        d.cancel()
        callAllSoonCalls(loop)
        self.assertEqual(cancelled.cancelled(), True)
        self.assertRaises(CancelledError, cancelled.result)
        self.failureResultOf(d).trap(CancelledError)