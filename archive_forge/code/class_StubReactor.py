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
class StubReactor:
    """
    A reactor stub which contains enough functionality to be used with the
    L{_Janitor}.

    @ivar iterations: A list of the arguments passed to L{iterate}.
    @ivar removeAllCalled: Number of times that L{removeAll} was called.
    @ivar selectables: The value that will be returned from L{removeAll}.
    @ivar delayedCalls: The value to return from L{getDelayedCalls}.
    """

    def __init__(self, delayedCalls: list[DelayedCall], selectables: list[object] | None=None) -> None:
        """
        @param delayedCalls: See L{StubReactor.delayedCalls}.
        @param selectables: See L{StubReactor.selectables}.
        """
        self.delayedCalls = delayedCalls
        self.iterations: list[float | None] = []
        self.removeAllCalled = 0
        if not selectables:
            selectables = []
        self.selectables = selectables

    def iterate(self, timeout: float | None=None) -> None:
        """
        Increment C{self.iterations}.
        """
        self.iterations.append(timeout)

    def getDelayedCalls(self) -> list[DelayedCall]:
        """
        Return C{self.delayedCalls}.
        """
        return self.delayedCalls

    def removeAll(self) -> list[object]:
        """
        Increment C{self.removeAllCalled} and return C{self.selectables}.
        """
        self.removeAllCalled += 1
        return self.selectables