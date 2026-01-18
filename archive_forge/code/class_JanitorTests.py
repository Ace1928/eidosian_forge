from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
class JanitorTests(SynchronousTestCase):
    """
    Tests for L{_Janitor}!
    """

    def test_cleanPendingSpinsReactor(self) -> None:
        """
        During pending-call cleanup, the reactor will be spun twice with an
        instant timeout. This is not a requirement, it is only a test for
        current behavior. Hopefully Trial will eventually not do this kind of
        reactor stuff.
        """
        reactor = StubReactor([])
        jan = _Janitor(None, None, reactor=reactor)
        jan._cleanPending()
        self.assertEqual(reactor.iterations, [0, 0])

    def test_cleanPendingCancelsCalls(self) -> None:
        """
        During pending-call cleanup, the janitor cancels pending timed calls.
        """

        def func() -> str:
            return 'Lulz'
        cancelled: list[DelayedCall] = []
        delayedCall = DelayedCall(300, func, (), {}, cancelled.append, lambda x: None)
        reactor = StubReactor([delayedCall])
        jan = _Janitor(None, None, reactor=reactor)
        jan._cleanPending()
        self.assertEqual(cancelled, [delayedCall])

    def test_cleanPendingReturnsDelayedCallStrings(self) -> None:
        """
        The Janitor produces string representations of delayed calls from the
        delayed call cleanup method. It gets the string representations
        *before* cancelling the calls; this is important because cancelling the
        call removes critical debugging information from the string
        representation.
        """
        delayedCall = DelayedCall(300, lambda: None, (), {}, lambda x: None, lambda x: None, seconds=lambda: 0)
        delayedCallString = str(delayedCall)
        reactor = StubReactor([delayedCall])
        jan = _Janitor(None, None, reactor=reactor)
        strings = jan._cleanPending()
        self.assertEqual(strings, [delayedCallString])

    def test_cleanReactorRemovesSelectables(self) -> None:
        """
        The Janitor will remove selectables during reactor cleanup.
        """
        reactor = StubReactor([])
        jan = _Janitor(None, None, reactor=reactor)
        jan._cleanReactor()
        self.assertEqual(reactor.removeAllCalled, 1)

    def test_cleanReactorKillsProcesses(self) -> None:
        """
        The Janitor will kill processes during reactor cleanup.
        """

        @implementer(IProcessTransport)
        class StubProcessTransport:
            """
            A stub L{IProcessTransport} provider which records signals.
            @ivar signals: The signals passed to L{signalProcess}.
            """

            def __init__(self) -> None:
                self.signals: list[str | int] = []

            def signalProcess(self, signal: str | int) -> None:
                """
                Append C{signal} to C{self.signals}.
                """
                self.signals.append(signal)
        pt = StubProcessTransport()
        reactor = StubReactor([], [pt])
        jan = _Janitor(None, None, reactor=reactor)
        jan._cleanReactor()
        self.assertEqual(pt.signals, ['KILL'])

    def test_cleanReactorReturnsSelectableStrings(self) -> None:
        """
        The Janitor returns string representations of the selectables that it
        cleaned up from the reactor cleanup method.
        """

        class Selectable:
            """
            A stub Selectable which only has an interesting string
            representation.
            """

            def __repr__(self) -> str:
                return '(SELECTABLE!)'
        reactor = StubReactor([], [Selectable()])
        jan = _Janitor(None, None, reactor=reactor)
        self.assertEqual(jan._cleanReactor(), ['(SELECTABLE!)'])

    def test_postCaseCleanupNoErrors(self) -> None:
        """
        The post-case cleanup method will return True and not call C{addError}
        on the result if there are no pending calls.
        """
        reactor = StubReactor([])
        test = object()
        reporter = StubErrorReporter()
        jan = _Janitor(test, reporter, reactor=reactor)
        self.assertTrue(jan.postCaseCleanup())
        self.assertEqual(reporter.errors, [])

    def test_postCaseCleanupWithErrors(self) -> None:
        """
        The post-case cleanup method will return False and call C{addError} on
        the result with a L{DirtyReactorAggregateError} Failure if there are
        pending calls.
        """
        delayedCall = DelayedCall(300, lambda: None, (), {}, lambda x: None, lambda x: None, seconds=lambda: 0)
        delayedCallString = str(delayedCall)
        reactor = StubReactor([delayedCall], [])
        test = object()
        reporter = StubErrorReporter()
        jan = _Janitor(test, reporter, reactor=reactor)
        self.assertFalse(jan.postCaseCleanup())
        self.assertEqual(len(reporter.errors), 1)
        self.assertEqual(reporter.errors[0][1].value.delayedCalls, [delayedCallString])

    def test_postClassCleanupNoErrors(self) -> None:
        """
        The post-class cleanup method will not call C{addError} on the result
        if there are no pending calls or selectables.
        """
        reactor = StubReactor([])
        test = object()
        reporter = StubErrorReporter()
        jan = _Janitor(test, reporter, reactor=reactor)
        jan.postClassCleanup()
        self.assertEqual(reporter.errors, [])

    def test_postClassCleanupWithPendingCallErrors(self) -> None:
        """
        The post-class cleanup method call C{addError} on the result with a
        L{DirtyReactorAggregateError} Failure if there are pending calls.
        """
        delayedCall = DelayedCall(300, lambda: None, (), {}, lambda x: None, lambda x: None, seconds=lambda: 0)
        delayedCallString = str(delayedCall)
        reactor = StubReactor([delayedCall], [])
        test = object()
        reporter = StubErrorReporter()
        jan = _Janitor(test, reporter, reactor=reactor)
        jan.postClassCleanup()
        self.assertEqual(len(reporter.errors), 1)
        self.assertEqual(reporter.errors[0][1].value.delayedCalls, [delayedCallString])

    def test_postClassCleanupWithSelectableErrors(self) -> None:
        """
        The post-class cleanup method call C{addError} on the result with a
        L{DirtyReactorAggregateError} Failure if there are selectables.
        """
        selectable = 'SELECTABLE HERE'
        reactor = StubReactor([], [selectable])
        test = object()
        reporter = StubErrorReporter()
        jan = _Janitor(test, reporter, reactor=reactor)
        jan.postClassCleanup()
        self.assertEqual(len(reporter.errors), 1)
        self.assertEqual(reporter.errors[0][1].value.selectables, [repr(selectable)])