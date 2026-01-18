import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
class Test_OldTestResult(unittest.TestCase):

    def assertOldResultWarning(self, test, failures):
        with warnings_helper.check_warnings(('TestResult has no add.+ method,', RuntimeWarning)):
            result = OldResult()
            test.run(result)
            self.assertEqual(len(result.failures), failures)

    def testOldTestResult(self):

        class Test(unittest.TestCase):

            def testSkip(self):
                self.skipTest('foobar')

            @unittest.expectedFailure
            def testExpectedFail(self):
                raise TypeError

            @unittest.expectedFailure
            def testUnexpectedSuccess(self):
                pass
        for test_name, should_pass in (('testSkip', True), ('testExpectedFail', True), ('testUnexpectedSuccess', False)):
            test = Test(test_name)
            self.assertOldResultWarning(test, int(not should_pass))

    def testOldTestTesultSetup(self):

        class Test(unittest.TestCase):

            def setUp(self):
                self.skipTest('no reason')

            def testFoo(self):
                pass
        self.assertOldResultWarning(Test('testFoo'), 0)

    def testOldTestResultClass(self):

        @unittest.skip('no reason')
        class Test(unittest.TestCase):

            def testFoo(self):
                pass
        self.assertOldResultWarning(Test('testFoo'), 0)

    def testOldResultWithRunner(self):

        class Test(unittest.TestCase):

            def testFoo(self):
                pass
        runner = unittest.TextTestRunner(resultclass=OldResult, stream=io.StringIO())
        runner.run(Test('testFoo'))