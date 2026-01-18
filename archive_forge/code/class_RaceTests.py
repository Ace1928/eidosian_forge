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
class RaceTests(unittest.SynchronousTestCase):
    """
    Tests for L{race}.
    """

    @given(beforeWinner=integers(min_value=0, max_value=3), afterWinner=integers(min_value=0, max_value=3))
    def test_success(self, beforeWinner: int, afterWinner: int) -> None:
        """
        When one of the L{Deferred}s passed to L{race} fires successfully,
        the L{Deferred} return by L{race} fires with the index of that
        L{Deferred} and its result and cancels the rest of the L{Deferred}s.

        @param beforeWinner: A randomly selected number of Deferreds to
            appear before the "winning" Deferred in the list passed in.

        @param beforeWinner: A randomly selected number of Deferreds to
            appear after the "winning" Deferred in the list passed in.
        """
        cancelledState, ds = _setupRaceState(beforeWinner + 1 + afterWinner)
        raceResult = race(ds)
        expected = object()
        ds[beforeWinner].callback(expected)
        assert_that(self.successResultOf(raceResult), equal_to((beforeWinner, expected)))
        expectedCancelledState = [1] * beforeWinner + [0] + [1] * afterWinner
        assert_that(cancelledState, equal_to(expectedCancelledState))

    @given(beforeWinner=integers(min_value=0, max_value=3), afterWinner=integers(min_value=0, max_value=3))
    def test_failure(self, beforeWinner: int, afterWinner: int) -> None:
        """
        When all of the L{Deferred}s passed to L{race} fire with failures,
        the L{Deferred} return by L{race} fires with L{FailureGroup} wrapping
        all of their failures.

        @param beforeWinner: A randomly selected number of Deferreds to
            appear before the "winning" Deferred in the list passed in.

        @param beforeWinner: A randomly selected number of Deferreds to
            appear after the "winning" Deferred in the list passed in.
        """
        cancelledState, ds = _setupRaceState(beforeWinner + 1 + afterWinner)
        failure = Failure(Exception('The test demands failures.'))
        raceResult = race(ds)
        for d in ds:
            d.errback(failure)
        actualFailure = self.failureResultOf(raceResult, FailureGroup)
        assert_that(actualFailure.value.failures, equal_to([failure] * len(ds)))
        assert_that(cancelledState, equal_to([0] * len(ds)))

    @given(beforeWinner=integers(min_value=0, max_value=3), afterWinner=integers(min_value=0, max_value=3))
    def test_resultAfterCancel(self, beforeWinner: int, afterWinner: int) -> None:
        """
        If one of the Deferreds fires after it was cancelled its result
        goes nowhere.  In particular, it does not cause any errors to be
        logged.
        """
        ds: list[Deferred[None]] = [Deferred() for n in range(beforeWinner + 2 + afterWinner)]
        raceResult = race(ds)
        ds[beforeWinner].callback(None)
        ds[beforeWinner + 1].callback(None)
        self.successResultOf(raceResult)
        assert_that(self.flushLoggedErrors(), empty())

    def test_resultFromCancel(self) -> None:
        """
        If one of the input Deferreds has a cancel function that fires it
        with success, nothing bad happens.
        """
        winner: Deferred[object] = Deferred()
        ds: list[Deferred[object]] = [winner, Deferred(canceller=lambda d: d.callback(object()))]
        expected = object()
        raceResult = race(ds)
        winner.callback(expected)
        assert_that(self.successResultOf(raceResult), equal_to((0, expected)))

    @given(numDeferreds=integers(min_value=1, max_value=3))
    def test_cancel(self, numDeferreds: int) -> None:
        """
        If the result of L{race} is cancelled then all of the L{Deferred}s
        passed in are cancelled.
        """
        cancelledState, ds = _setupRaceState(numDeferreds)
        raceResult = race(ds)
        raceResult.cancel()
        assert_that(cancelledState, equal_to([1] * numDeferreds))
        self.failureResultOf(raceResult, FailureGroup)