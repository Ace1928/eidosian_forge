from __future__ import annotations
from os import environ
from datetime import datetime, timedelta
from time import mktime as mktime_real
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.trial.unittest import SkipTest, TestCase
def addTZCleanup(testCase: TestCase) -> None:
    """
    Add cleanup hooks to a test case to reset timezone to original value.

    @param testCase: the test case to add the cleanup to.
    @type testCase: L{unittest.TestCase}
    """
    tzIn = environ.get('TZ', None)

    @testCase.addCleanup
    def resetTZ() -> None:
        setTZ(tzIn)