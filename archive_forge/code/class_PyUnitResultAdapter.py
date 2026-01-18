import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
class PyUnitResultAdapter:
    """
    Wrap a C{TestResult} from the standard library's C{unittest} so that it
    supports the extended result types from Trial, and also supports
    L{twisted.python.failure.Failure}s being passed to L{addError} and
    L{addFailure}.
    """

    def __init__(self, original):
        """
        @param original: A C{TestResult} instance from C{unittest}.
        """
        self.original = original

    def _exc_info(self, err):
        return util.excInfoOrFailureToExcInfo(err)

    def startTest(self, method):
        self.original.startTest(method)

    def stopTest(self, method):
        self.original.stopTest(method)

    def addFailure(self, test, fail):
        self.original.addFailure(test, self._exc_info(fail))

    def addError(self, test, error):
        self.original.addError(test, self._exc_info(error))

    def _unsupported(self, test, feature, info):
        self.original.addFailure(test, (UnsupportedTrialFeature, UnsupportedTrialFeature(feature, info), None))

    def addSkip(self, test, reason):
        """
        Report the skip as a failure.
        """
        self.original.addSkip(test, reason)

    def addUnexpectedSuccess(self, test, todo=None):
        """
        Report the unexpected success as a failure.
        """
        self._unsupported(test, 'unexpected success', todo)

    def addExpectedFailure(self, test, error):
        """
        Report the expected failure (i.e. todo) as a failure.
        """
        self._unsupported(test, 'expected failure', error)

    def addSuccess(self, test):
        self.original.addSuccess(test)

    def upDownError(self, method, error, warn, printStatus):
        pass