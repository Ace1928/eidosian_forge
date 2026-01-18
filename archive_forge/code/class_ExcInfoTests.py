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
class ExcInfoTests(SynchronousTestCase):
    """
    Tests for L{excInfoOrFailureToExcInfo}.
    """

    def test_excInfo(self) -> None:
        """
        L{excInfoOrFailureToExcInfo} returns exactly what it is passed, if it is
        passed a tuple like the one returned by L{sys.exc_info}.
        """
        info = (ValueError, ValueError('foo'), None)
        self.assertTrue(info is excInfoOrFailureToExcInfo(info))

    def test_failure(self) -> None:
        """
        When called with a L{Failure} instance, L{excInfoOrFailureToExcInfo}
        returns a tuple like the one returned by L{sys.exc_info}, with the
        elements taken from the type, value, and traceback of the failure.
        """
        try:
            1 / 0
        except BaseException:
            f = Failure()
        self.assertEqual((f.type, f.value, f.tb), excInfoOrFailureToExcInfo(f))