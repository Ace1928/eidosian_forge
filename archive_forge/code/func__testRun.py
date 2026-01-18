import unittest as pyunit
from twisted.trial import itrial, reporter, runner, unittest
from twisted.trial.test import mockdoctest
def _testRun(self, suite: pyunit.TestSuite) -> None:
    """
        Run C{suite} and check the result.
        """
    result = reporter.TestResult()
    suite.run(result)
    self.assertEqual(5, result.successes)
    self.assertEqual(2, len(result.failures))