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
class DeferredCancellerTests(unittest.SynchronousTestCase):

    def setUp(self) -> None:
        self.callbackResults: Optional[str] = None
        self.errbackResults: Optional[Failure] = None
        self.callback2Results: Optional[str] = None
        self.cancellerCallCount = 0

    def tearDown(self) -> None:
        self.assertIn(self.cancellerCallCount, (0, 1))

    def _callback(self, data: str) -> str:
        self.callbackResults = data
        return data

    def _callback2(self, data: str) -> None:
        self.callback2Results = data

    def _errback(self, error: Failure) -> None:
        self.errbackResults = error

    def test_noCanceller(self) -> None:
        """
        A L{Deferred} without a canceller must errback with a
        L{defer.CancelledError} and not callback.
        """
        d: Deferred[None] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        assert self.errbackResults is not None
        self.assertEqual(self.errbackResults.type, defer.CancelledError)
        self.assertIsNone(self.callbackResults)

    def test_raisesAfterCancelAndCallback(self) -> None:
        """
        A L{Deferred} without a canceller, when cancelled must allow
        a single extra call to callback, and raise
        L{defer.AlreadyCalledError} if callbacked or errbacked thereafter.
        """
        d: Deferred[None] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        d.callback(None)
        self.assertRaises(defer.AlreadyCalledError, d.callback, None)
        self.assertRaises(defer.AlreadyCalledError, d.errback, Exception())

    def test_raisesAfterCancelAndErrback(self) -> None:
        """
        A L{Deferred} without a canceller, when cancelled must allow
        a single extra call to errback, and raise
        L{defer.AlreadyCalledError} if callbacked or errbacked thereafter.
        """
        d: Deferred[None] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        d.errback(Exception())
        self.assertRaises(defer.AlreadyCalledError, d.callback, None)
        self.assertRaises(defer.AlreadyCalledError, d.errback, Exception())

    def test_noCancellerMultipleCancelsAfterCancelAndCallback(self) -> None:
        """
        A L{Deferred} without a canceller, when cancelled and then
        callbacked, ignores multiple cancels thereafter.
        """
        d: Deferred[None] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        currentFailure = self.errbackResults
        d.callback(None)
        d.cancel()
        self.assertIs(currentFailure, self.errbackResults)

    def test_noCancellerMultipleCancelsAfterCancelAndErrback(self) -> None:
        """
        A L{Deferred} without a canceller, when cancelled and then
        errbacked, ignores multiple cancels thereafter.
        """
        d: Deferred[None] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        assert self.errbackResults is not None
        self.assertEqual(self.errbackResults.type, defer.CancelledError)
        currentFailure = self.errbackResults
        d.errback(GenericError())
        self.assertEqual(self.errbackResults.type, defer.CancelledError)
        d.cancel()
        self.assertIs(currentFailure, self.errbackResults)

    def test_noCancellerMultipleCancel(self) -> None:
        """
        Calling cancel multiple times on a deferred with no canceller
        results in a L{defer.CancelledError}. Subsequent calls to cancel
        do not cause an error.
        """
        d: Deferred[None] = Deferred()
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        assert self.errbackResults is not None
        self.assertEqual(self.errbackResults.type, defer.CancelledError)
        currentFailure = self.errbackResults
        d.cancel()
        self.assertIs(currentFailure, self.errbackResults)

    def test_cancellerMultipleCancel(self) -> None:
        """
        Verify that calling cancel multiple times on a deferred with a
        canceller that does not errback results in a
        L{defer.CancelledError} and that subsequent calls to cancel do not
        cause an error and that after all that, the canceller was only
        called once.
        """

        def cancel(d: Deferred[object]) -> None:
            self.cancellerCallCount += 1
        d: Deferred[None] = Deferred(canceller=cancel)
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        assert self.errbackResults is not None
        self.assertEqual(self.errbackResults.type, defer.CancelledError)
        currentFailure = self.errbackResults
        d.cancel()
        self.assertIs(currentFailure, self.errbackResults)
        self.assertEqual(self.cancellerCallCount, 1)

    def test_simpleCanceller(self) -> None:
        """
        Verify that a L{Deferred} calls its specified canceller when
        it is cancelled, and that further call/errbacks raise
        L{defer.AlreadyCalledError}.
        """

        def cancel(d: Deferred[object]) -> None:
            self.cancellerCallCount += 1
        d: Deferred[None] = Deferred(canceller=cancel)
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        self.assertEqual(self.cancellerCallCount, 1)
        assert self.errbackResults is not None
        self.assertEqual(self.errbackResults.type, defer.CancelledError)
        self.assertRaises(defer.AlreadyCalledError, d.callback, None)
        self.assertRaises(defer.AlreadyCalledError, d.errback, Exception())

    def test_cancellerArg(self) -> None:
        """
        Verify that a canceller is given the correct deferred argument.
        """

        def cancel(d1: Deferred[object]) -> None:
            self.assertIs(d1, d)
        d: Deferred[None] = Deferred(canceller=cancel)
        d.addCallbacks(self._callback, self._errback)
        d.cancel()

    def test_cancelAfterCallback(self) -> None:
        """
        Test that cancelling a deferred after it has been callbacked does
        not cause an error.
        """

        def cancel(d: Deferred[object]) -> None:
            self.cancellerCallCount += 1
            d.errback(GenericError())
        d: Deferred[str] = Deferred(canceller=cancel)
        d.addCallbacks(self._callback, self._errback)
        d.callback('biff!')
        d.cancel()
        self.assertEqual(self.cancellerCallCount, 0)
        self.assertIsNone(self.errbackResults)
        self.assertEqual(self.callbackResults, 'biff!')

    def test_cancelAfterErrback(self) -> None:
        """
        Test that cancelling a L{Deferred} after it has been errbacked does
        not result in a L{defer.CancelledError}.
        """

        def cancel(d: Deferred[object]) -> None:
            self.cancellerCallCount += 1
            d.errback(GenericError())
        d: Deferred[None] = Deferred(canceller=cancel)
        d.addCallbacks(self._callback, self._errback)
        d.errback(GenericError())
        d.cancel()
        self.assertEqual(self.cancellerCallCount, 0)
        assert self.errbackResults is not None
        self.assertEqual(self.errbackResults.type, GenericError)
        self.assertIsNone(self.callbackResults)

    def test_cancellerThatErrbacks(self) -> None:
        """
        Test a canceller which errbacks its deferred.
        """

        def cancel(d: Deferred[object]) -> None:
            self.cancellerCallCount += 1
            d.errback(GenericError())
        d: Deferred[None] = Deferred(canceller=cancel)
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        self.assertEqual(self.cancellerCallCount, 1)
        assert self.errbackResults is not None
        self.assertEqual(self.errbackResults.type, GenericError)

    def test_cancellerThatCallbacks(self) -> None:
        """
        Test a canceller which calls its deferred.
        """

        def cancel(d: Deferred[object]) -> None:
            self.cancellerCallCount += 1
            d.callback('hello!')
        d: Deferred[None] = Deferred(canceller=cancel)
        d.addCallbacks(self._callback, self._errback)
        d.cancel()
        self.assertEqual(self.cancellerCallCount, 1)
        self.assertEqual(self.callbackResults, 'hello!')
        self.assertIsNone(self.errbackResults)

    def test_cancelNestedDeferred(self) -> None:
        """
        Verify that a Deferred, a, which is waiting on another Deferred, b,
        returned from one of its callbacks, will propagate
        L{defer.CancelledError} when a is cancelled.
        """

        def innerCancel(d: Deferred[object]) -> None:
            self.cancellerCallCount += 1

        def cancel(d: Deferred[object]) -> None:
            self.assertTrue(False)
        b: Deferred[None] = Deferred(canceller=innerCancel)
        a: Deferred[None] = Deferred(canceller=cancel)
        a.callback(None)
        a.addCallback(lambda data: b)
        a.cancel()
        a.addCallbacks(self._callback, self._errback)
        self.assertEqual(self.cancellerCallCount, 1)
        assert self.errbackResults is not None
        self.assertEqual(self.errbackResults.type, defer.CancelledError)