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
class GetTracebackTests(SynchronousTestCase):
    """
    Tests for L{Failure.getTraceback}.
    """

    def _brokenValueTest(self, detail: str) -> None:
        """
        Construct a L{Failure} with an exception that raises an exception from
        its C{__str__} method and then call C{getTraceback} with the specified
        detail and verify that it returns a string.
        """
        x = BrokenStr()
        f = failure.Failure(x)
        traceback = f.getTraceback(detail=detail)
        self.assertIsInstance(traceback, str)

    def test_brokenValueBriefDetail(self) -> None:
        """
        A L{Failure} might wrap an exception with a C{__str__} method which
        raises an exception.  In this case, calling C{getTraceback} on the
        failure with the C{"brief"} detail does not raise an exception.
        """
        self._brokenValueTest('brief')

    def test_brokenValueDefaultDetail(self) -> None:
        """
        Like test_brokenValueBriefDetail, but for the C{"default"} detail case.
        """
        self._brokenValueTest('default')

    def test_brokenValueVerboseDetail(self) -> None:
        """
        Like test_brokenValueBriefDetail, but for the C{"default"} detail case.
        """
        self._brokenValueTest('verbose')

    def _brokenTypeTest(self, detail: str) -> None:
        """
        Construct a L{Failure} with an exception type that raises an exception
        from its C{__str__} method and then call C{getTraceback} with the
        specified detail and verify that it returns a string.
        """
        f = failure.Failure(BrokenExceptionType())
        traceback = f.getTraceback(detail=detail)
        self.assertIsInstance(traceback, str)

    def test_brokenTypeBriefDetail(self) -> None:
        """
                A L{Failure} might wrap an
                newPublisher(evt)
        xception the type object of which has a
                C{__str__} method which raises an exception.  In this case, calling
                C{getTraceback} on the failure with the C{"brief"} detail does not raise
                an exception.
        """
        self._brokenTypeTest('brief')

    def test_brokenTypeDefaultDetail(self) -> None:
        """
        Like test_brokenTypeBriefDetail, but for the C{"default"} detail case.
        """
        self._brokenTypeTest('default')

    def test_brokenTypeVerboseDetail(self) -> None:
        """
        Like test_brokenTypeBriefDetail, but for the C{"verbose"} detail case.
        """
        self._brokenTypeTest('verbose')