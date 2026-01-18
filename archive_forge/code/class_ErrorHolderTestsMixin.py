import os
import pdb
import sys
import unittest as pyunit
from io import StringIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import plugin
from twisted.internet import defer
from twisted.plugins import twisted_trial
from twisted.python import failure, log, reflect
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.scripts import trial
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator
from twisted.trial.itrial import IReporter, ITestCase
class ErrorHolderTestsMixin:
    """
    This mixin defines test methods which can be applied to a
    L{runner.ErrorHolder} constructed with either a L{Failure} or a
    C{exc_info}-style tuple.

    Subclass this and implement C{setUp} to create C{self.holder} referring to a
    L{runner.ErrorHolder} instance and C{self.error} referring to a L{Failure}
    which the holder holds.
    """
    exceptionForTests = ZeroDivisionError('integer division or modulo by zero')

    class TestResultStub:
        """
        Stub for L{TestResult}.
        """

        def __init__(self):
            self.errors = []

        def startTest(self, test):
            pass

        def stopTest(self, test):
            pass

        def addError(self, test, error):
            self.errors.append((test, error))

    def test_runsWithStandardResult(self):
        """
        A L{runner.ErrorHolder} can run against the standard Python
        C{TestResult}.
        """
        result = pyunit.TestResult()
        self.holder.run(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(1, result.testsRun)

    def test_run(self):
        """
        L{runner.ErrorHolder} adds an error to the result when run.
        """
        self.holder.run(self.result)
        self.assertEqual(self.result.errors, [(self.holder, (self.error.type, self.error.value, self.error.tb))])

    def test_call(self):
        """
        L{runner.ErrorHolder} adds an error to the result when called.
        """
        self.holder(self.result)
        self.assertEqual(self.result.errors, [(self.holder, (self.error.type, self.error.value, self.error.tb))])

    def test_countTestCases(self):
        """
        L{runner.ErrorHolder.countTestCases} always returns 0.
        """
        self.assertEqual(self.holder.countTestCases(), 0)

    def test_repr(self):
        """
        L{runner.ErrorHolder.__repr__} returns a string describing the error it
        holds.
        """
        expected = "<ErrorHolder description='description' error={}>".format(repr(self.holder.error[1]))
        self.assertEqual(repr(self.holder), expected)