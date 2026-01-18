import time
from twisted.internet import reactor, task
from twisted.python import failure, log
from twisted.trial import _synctest, reporter, unittest
class AsynchronousLogErrorsTests(LogErrorsMixin, unittest.TestCase):
    MockTest = Mask.AsynchronousFailureLogging

    def test_inCallback(self):
        """
        Test that errors logged in callbacks get reported as test errors.
        """
        test = self.MockTest('test_inCallback')
        test(self.result)
        self.assertEqual(len(self.result.errors), 1)
        self.assertTrue(self.result.errors[0][1].check(ZeroDivisionError), self.result.errors[0][1])