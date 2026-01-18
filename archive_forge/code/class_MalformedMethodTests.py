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
class MalformedMethodTests(unittest.SynchronousTestCase):
    """
    Test that trial manages when test methods don't have correct signatures.
    """

    class ContainMalformed(pyunit.TestCase):
        """
        This TestCase holds malformed test methods that trial should handle.
        """

        def test_foo(self, blah):
            pass

        def test_bar():
            pass
        test_spam = defer.inlineCallbacks(test_bar)

    def _test(self, method):
        """
        Wrapper for one of the test method of L{ContainMalformed}.
        """
        stream = StringIO()
        trialRunner = runner.TrialRunner(reporter.Reporter, stream=stream)
        test = MalformedMethodTests.ContainMalformed(method)
        result = trialRunner.run(test)
        self.assertEqual(result.testsRun, 1)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(len(result.errors), 1)

    def test_extraArg(self):
        """
        Test when the method has extra (useless) arguments.
        """
        self._test('test_foo')

    def test_noArg(self):
        """
        Test when the method doesn't have even self as argument.
        """
        self._test('test_bar')

    def test_decorated(self):
        """
        Test a decorated method also fails.
        """
        self._test('test_spam')