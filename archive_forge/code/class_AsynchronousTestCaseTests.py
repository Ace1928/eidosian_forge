from twisted.trial.unittest import SynchronousTestCase, TestCase
class AsynchronousTestCaseTests(TestCaseMixin, SynchronousTestCase):

    class MyTestCase(TestCase):
        """
        Some test methods which can be used to test behaviors of
        L{TestCase}.
        """

        def test_1(self):
            pass