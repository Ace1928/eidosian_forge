from __future__ import annotations
import linecache
import pdb
import re
import sys
import traceback
from dis import distb
from io import StringIO
from traceback import FrameSummary
from types import TracebackType
from typing import Any, Generator
from unittest import skipIf
from cython_test_exception_raiser import raiser
from twisted.python import failure, reflect
from twisted.trial.unittest import SynchronousTestCase
class FindFailureTests(SynchronousTestCase):
    """
    Tests for functionality related to L{Failure._findFailure}.
    """

    def test_findNoFailureInExceptionHandler(self) -> None:
        """
        Within an exception handler, _findFailure should return
        L{None} in case no Failure is associated with the current
        exception.
        """
        try:
            1 / 0
        except BaseException:
            self.assertIsNone(failure.Failure._findFailure())
        else:
            self.fail('No exception raised from 1/0!?')

    def test_findNoFailure(self) -> None:
        """
        Outside of an exception handler, _findFailure should return None.
        """
        self.assertIsNone(sys.exc_info()[-1])
        self.assertIsNone(failure.Failure._findFailure())

    def test_findFailure(self) -> None:
        """
        Within an exception handler, it should be possible to find the
        original Failure that caused the current exception (if it was
        caused by raiseException).
        """
        f = getDivisionFailure()
        f.cleanFailure()
        try:
            f.raiseException()
        except BaseException:
            self.assertEqual(failure.Failure._findFailure(), f)
        else:
            self.fail('No exception raised from raiseException!?')

    def test_failureConstructionFindsOriginalFailure(self) -> None:
        """
        When a Failure is constructed in the context of an exception
        handler that is handling an exception raised by
        raiseException, the new Failure should be chained to that
        original Failure.
        Means the new failure should still show the same origin frame,
        but with different complete stack trace (as not thrown at same place).
        """
        f = getDivisionFailure()
        f.cleanFailure()
        try:
            f.raiseException()
        except BaseException:
            newF = failure.Failure()
            tb = f.getTraceback().splitlines()
            new_tb = newF.getTraceback().splitlines()
            self.assertNotEqual(tb, new_tb)
            self.assertEqual(tb[-3:], new_tb[-3:])
        else:
            self.fail('No exception raised from raiseException!?')

    @skipIf(raiser is None, 'raiser extension not available')
    def test_failureConstructionWithMungedStackSucceeds(self) -> None:
        """
        Pyrex and Cython are known to insert fake stack frames so as to give
        more Python-like tracebacks. These stack frames with empty code objects
        should not break extraction of the exception.
        """
        try:
            raiser.raiseException()
        except raiser.RaiserException:
            f = failure.Failure()
            self.assertTrue(f.check(raiser.RaiserException))
        else:
            self.fail('No exception raised from extension?!')