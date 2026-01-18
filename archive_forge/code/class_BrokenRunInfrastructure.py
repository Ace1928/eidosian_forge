import unittest
from sys import exc_info
from twisted.python.failure import Failure
class BrokenRunInfrastructure(unittest.TestCase):
    """
    A test suite that is broken at the level of integration between
    L{TestCase.run} and the results object.
    """

    def run(self, result):
        """
        Override the normal C{run} behavior to pass the result object
        along to the test method.  Each test method needs the result object so
        that it can implement its particular kind of brokenness.
        """
        return getattr(self, self._testMethodName)(result)

    def test_addSuccess(self, result):
        """
        Violate the L{TestResult.addSuccess} interface.
        """
        result.addSuccess(_NonStringId())

    def test_addError(self, result):
        """
        Violate the L{TestResult.addError} interface.
        """
        try:
            raise Exception('test_addError')
        except BaseException:
            err = exc_info()
        result.addError(_NonStringId(), err)

    def test_addFailure(self, result):
        """
        Violate the L{TestResult.addFailure} interface.
        """
        try:
            raise Exception('test_addFailure')
        except BaseException:
            err = exc_info()
        result.addFailure(_NonStringId(), err)

    def test_addSkip(self, result):
        """
        Violate the L{TestResult.addSkip} interface.
        """
        result.addSkip(_NonStringId(), 'test_addSkip')

    def test_addExpectedFailure(self, result):
        """
        Violate the L{TestResult.addExpectedFailure} interface.
        """
        try:
            raise Exception('test_addExpectedFailure')
        except BaseException:
            err = Failure()
        result.addExpectedFailure(_NonStringId(), err)

    def test_addUnexpectedSuccess(self, result):
        """
        Violate the L{TestResult.addUnexpectedSuccess} interface.
        """
        result.addUnexpectedSuccess(_NonStringId())