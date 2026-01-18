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
class TestHolderTests(unittest.SynchronousTestCase):

    def setUp(self):
        self.description = 'description'
        self.holder = runner.TestHolder(self.description)

    def test_holder(self):
        """
        Check that L{runner.TestHolder} takes a description as a parameter
        and that this description is returned by the C{id} and
        C{shortDescription} methods.
        """
        self.assertEqual(self.holder.id(), self.description)
        self.assertEqual(self.holder.shortDescription(), self.description)

    def test_holderImplementsITestCase(self):
        """
        L{runner.TestHolder} implements L{ITestCase}.
        """
        self.assertIdentical(self.holder, ITestCase(self.holder))
        self.assertTrue(verifyObject(ITestCase, self.holder), '%r claims to provide %r but does not do so correctly.' % (self.holder, ITestCase))

    def test_runsWithStandardResult(self):
        """
        A L{runner.TestHolder} can run against the standard Python
        C{TestResult}.
        """
        result = pyunit.TestResult()
        self.holder.run(result)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(1, result.testsRun)