import unittest
from sys import exc_info
from twisted.python.failure import Failure
class PyUnitTest(unittest.TestCase):

    def test_pass(self):
        """
        A passing test.
        """

    def test_error(self):
        """
        A test which raises an exception to cause an error.
        """
        raise Exception('pyunit error')

    def test_fail(self):
        """
        A test which uses L{unittest.TestCase.fail} to cause a failure.
        """
        self.fail('pyunit failure')

    @unittest.skip('pyunit skip')
    def test_skip(self):
        """
        A test which uses the L{unittest.skip} decorator to cause a skip.
        """