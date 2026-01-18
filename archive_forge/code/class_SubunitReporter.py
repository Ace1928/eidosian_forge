from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
@implementer(itrial.IReporter)
class SubunitReporter:
    """
    Reports test output via Subunit.

    @ivar _subunit: The subunit protocol client that we are wrapping.

    @ivar _successful: An internal variable, used to track whether we have
        received only successful results.

    @since: 10.0
    """
    testsRun = None

    def __init__(self, stream=sys.stdout, tbformat='default', realtime=False, publisher=None):
        """
        Construct a L{SubunitReporter}.

        @param stream: A file-like object representing the stream to print
            output to. Defaults to stdout.
        @param tbformat: The format for tracebacks. Ignored, since subunit
            always uses Python's standard format.
        @param realtime: Whether or not to print exceptions in the middle
            of the test results. Ignored, since subunit always does this.
        @param publisher: The log publisher which will be preserved for
            reporting events. Ignored, as it's not relevant to subunit.
        """
        if TestProtocolClient is None:
            raise Exception('Subunit not available')
        self._subunit = TestProtocolClient(stream)
        self._successful = True

    def done(self):
        """
        Record that the entire test suite run is finished.

        We do nothing, since a summary clause is irrelevant to the subunit
        protocol.
        """
        pass

    @property
    def shouldStop(self):
        """
        Whether or not the test runner should stop running tests.
        """
        return self._subunit.shouldStop

    def stop(self):
        """
        Signal that the test runner should stop running tests.
        """
        return self._subunit.stop()

    def wasSuccessful(self):
        """
        Has the test run been successful so far?

        @return: C{True} if we have received no reports of errors or failures,
            C{False} otherwise.
        """
        return self._successful

    def startTest(self, test):
        """
        Record that C{test} has started.
        """
        return self._subunit.startTest(test)

    def stopTest(self, test):
        """
        Record that C{test} has completed.
        """
        return self._subunit.stopTest(test)

    def addSuccess(self, test):
        """
        Record that C{test} was successful.
        """
        return self._subunit.addSuccess(test)

    def addSkip(self, test, reason):
        """
        Record that C{test} was skipped for C{reason}.

        Some versions of subunit don't have support for addSkip. In those
        cases, the skip is reported as a success.

        @param test: A unittest-compatible C{TestCase}.
        @param reason: The reason for it being skipped. The C{str()} of this
            object will be included in the subunit output stream.
        """
        addSkip = getattr(self._subunit, 'addSkip', None)
        if addSkip is None:
            self.addSuccess(test)
        else:
            self._subunit.addSkip(test, reason)

    def addError(self, test, err):
        """
        Record that C{test} failed with an unexpected error C{err}.

        Also marks the run as being unsuccessful, causing
        L{SubunitReporter.wasSuccessful} to return C{False}.
        """
        self._successful = False
        return self._subunit.addError(test, util.excInfoOrFailureToExcInfo(err))

    def addFailure(self, test, err):
        """
        Record that C{test} failed an assertion with the error C{err}.

        Also marks the run as being unsuccessful, causing
        L{SubunitReporter.wasSuccessful} to return C{False}.
        """
        self._successful = False
        return self._subunit.addFailure(test, util.excInfoOrFailureToExcInfo(err))

    def addExpectedFailure(self, test, failure, todo=None):
        """
        Record an expected failure from a test.

        Some versions of subunit do not implement this. For those versions, we
        record a success.
        """
        failure = util.excInfoOrFailureToExcInfo(failure)
        addExpectedFailure = getattr(self._subunit, 'addExpectedFailure', None)
        if addExpectedFailure is None:
            self.addSuccess(test)
        else:
            addExpectedFailure(test, failure)

    def addUnexpectedSuccess(self, test, todo=None):
        """
        Record an unexpected success.

        Since subunit has no way of expressing this concept, we record a
        success on the subunit stream.
        """
        self.addSuccess(test)