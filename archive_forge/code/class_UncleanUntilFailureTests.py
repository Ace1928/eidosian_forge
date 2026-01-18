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
class UncleanUntilFailureTests(UntilFailureTests):
    """
    Test that the run-until-failure feature works correctly with the unclean
    error suppressor.
    """

    def setUp(self):
        UntilFailureTests.setUp(self)
        self.runner = runner.TrialRunner(reporter.Reporter, stream=self.stream, uncleanWarnings=True)

    def _getFailures(self, result):
        """
        Get the number of failures that were reported to a result that
        is wrapped in an UncleanFailureWrapper.
        """
        return len(result._originalReporter.failures)