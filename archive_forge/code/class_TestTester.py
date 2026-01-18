from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
class TestTester(unittest.TestCase):

    def getTest(self, name: str) -> pyunit.TestCase:
        raise NotImplementedError('must override me')

    def runTest(self, name: str) -> reporter.TestResult:
        result = reporter.TestResult()
        self.getTest(name).run(result)
        return result