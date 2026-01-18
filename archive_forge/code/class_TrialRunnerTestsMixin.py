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
class TrialRunnerTestsMixin:
    """
    Mixin defining tests for L{runner.TrialRunner}.
    """

    def test_empty(self):
        """
        Empty test method, used by the other tests.
        """

    def _getObservers(self):
        return log.theLogPublisher.observers

    def test_addObservers(self):
        """
        Any log system observers L{TrialRunner.run} adds are removed by the
        time it returns.
        """
        originalCount = len(self._getObservers())
        self.runner.run(self.test)
        newCount = len(self._getObservers())
        self.assertEqual(newCount, originalCount)

    def test_logFileAlwaysActive(self):
        """
        Test that a new file is opened on each run.
        """
        logPath = FilePath(self.runner.workingDirectory).child(self.runner.logfile)
        for i in range(2):
            self.runner.run(self.test)
            logPath.restat()
            self.assertTrue(logPath.exists())
            logPath.remove()