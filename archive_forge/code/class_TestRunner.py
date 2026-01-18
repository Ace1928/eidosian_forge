import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
class TestRunner(tests.TestCase):

    def dummy_test(self):
        pass

    def run_test_runner(self, testrunner, test):
        """Run suite in testrunner, saving global state and restoring it.

        This current saves and restores:
        TestCaseInTempDir.TEST_ROOT

        There should be no tests in this file that use
        breezy.tests.TextTestRunner without using this convenience method,
        because of our use of global state.
        """
        old_root = tests.TestCaseInTempDir.TEST_ROOT
        try:
            tests.TestCaseInTempDir.TEST_ROOT = None
            return testrunner.run(test)
        finally:
            tests.TestCaseInTempDir.TEST_ROOT = old_root

    def test_known_failure_failed_run(self):

        class Test(tests.TestCase):

            def known_failure_test(self):
                self.expectFailure('failed', self.assertTrue, False)
        test = unittest.TestSuite()
        test.addTest(Test('known_failure_test'))

        def failing_test():
            raise AssertionError('foo')
        test.addTest(unittest.FunctionTestCase(failing_test))
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream)
        self.run_test_runner(runner, test)
        self.assertContainsRe(stream.getvalue(), "(?sm)^brz selftest.*$.*^======================================================================\n^FAIL: failing_test\n^----------------------------------------------------------------------\nTraceback \\(most recent call last\\):\n  .*    raise AssertionError\\('foo'\\)\n.*^----------------------------------------------------------------------\n.*FAILED \\(failures=1, known_failure_count=1\\)")

    def test_known_failure_ok_run(self):

        class Test(tests.TestCase):

            def known_failure_test(self):
                self.knownFailure('Never works...')
        test = Test('known_failure_test')
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream)
        self.run_test_runner(runner, test)
        self.assertContainsRe(stream.getvalue(), '\n-*\nRan 1 test in .*\n\nOK \\(known_failures=1\\)\n')

    def test_unexpected_success_bad(self):

        class Test(tests.TestCase):

            def test_truth(self):
                self.expectFailure('No absolute truth', self.assertTrue, True)
        runner = tests.TextTestRunner(stream=StringIO())
        self.run_test_runner(runner, Test('test_truth'))
        self.assertContainsRe(runner.stream.getvalue(), '=+\nFAIL: \\S+\\.test_truth\n-+\n(?:.*\n)*\\s*(?:Text attachment: )?reason(?:\n-+\n|: {{{)No absolute truth(?:\n-+\n|}}}\n)(?:.*\n)*-+\nRan 1 test in .*\n\nFAILED \\(failures=1\\)\n\\Z')

    def test_result_decorator(self):
        calls = []

        class LoggingDecorator(ExtendedToOriginalDecorator):

            def startTest(self, test):
                ExtendedToOriginalDecorator.startTest(self, test)
                calls.append('start')
        test = unittest.FunctionTestCase(lambda: None)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream, result_decorators=[LoggingDecorator])
        self.run_test_runner(runner, test)
        self.assertLength(1, calls)

    def test_skipped_test(self):

        class SkippingTest(tests.TestCase):

            def skipping_test(self):
                raise tests.TestSkipped('test intentionally skipped')
        runner = tests.TextTestRunner(stream=StringIO())
        test = SkippingTest('skipping_test')
        result = self.run_test_runner(runner, test)
        self.assertTrue(result.wasSuccessful())

    def test_skipped_from_setup(self):
        calls = []

        class SkippedSetupTest(tests.TestCase):

            def setUp(self):
                calls.append('setUp')
                self.addCleanup(self.cleanup)
                raise tests.TestSkipped('skipped setup')

            def test_skip(self):
                self.fail('test reached')

            def cleanup(self):
                calls.append('cleanup')
        runner = tests.TextTestRunner(stream=StringIO())
        test = SkippedSetupTest('test_skip')
        result = self.run_test_runner(runner, test)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(['setUp', 'cleanup'], calls)

    def test_skipped_from_test(self):
        calls = []

        class SkippedTest(tests.TestCase):

            def setUp(self):
                super().setUp()
                calls.append('setUp')
                self.addCleanup(self.cleanup)

            def test_skip(self):
                raise tests.TestSkipped('skipped test')

            def cleanup(self):
                calls.append('cleanup')
        runner = tests.TextTestRunner(stream=StringIO())
        test = SkippedTest('test_skip')
        result = self.run_test_runner(runner, test)
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(['setUp', 'cleanup'], calls)

    def test_not_applicable(self):

        class Test(tests.TestCase):

            def not_applicable_test(self):
                raise tests.TestNotApplicable('this test never runs')
        out = StringIO()
        runner = tests.TextTestRunner(stream=out, verbosity=2)
        test = Test('not_applicable_test')
        result = self.run_test_runner(runner, test)
        self.log(out.getvalue())
        self.assertTrue(result.wasSuccessful())
        self.assertTrue(result.wasStrictlySuccessful())
        self.assertContainsRe(out.getvalue(), '(?m)not_applicable_test  * N/A')
        self.assertContainsRe(out.getvalue(), '(?m)^    this test never runs')

    def test_unsupported_features_listed(self):
        """When unsupported features are encountered they are detailed."""

        class Feature1(features.Feature):

            def _probe(self):
                return False

        class Feature2(features.Feature):

            def _probe(self):
                return False
        test1 = SampleTestCase('_test_pass')
        test1._test_needs_features = [Feature1()]
        test2 = SampleTestCase('_test_pass')
        test2._test_needs_features = [Feature2()]
        test = unittest.TestSuite()
        test.addTest(test1)
        test.addTest(test2)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream)
        self.run_test_runner(runner, test)
        lines = stream.getvalue().splitlines()
        self.assertEqual(['OK', "Missing feature 'Feature1' skipped 1 tests.", "Missing feature 'Feature2' skipped 1 tests."], lines[-3:])

    def test_verbose_test_count(self):
        """A verbose test run reports the right test count at the start"""
        suite = TestUtil.TestSuite([unittest.FunctionTestCase(lambda: None), unittest.FunctionTestCase(lambda: None)])
        self.assertEqual(suite.countTestCases(), 2)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream, verbosity=2)
        self.run_test_runner(runner, tests.CountingDecorator(suite))
        self.assertStartsWith(stream.getvalue(), 'running 2 tests')

    def test_startTestRun(self):
        """run should call result.startTestRun()"""
        calls = []

        class LoggingDecorator(ExtendedToOriginalDecorator):

            def startTestRun(self):
                ExtendedToOriginalDecorator.startTestRun(self)
                calls.append('startTestRun')
        test = unittest.FunctionTestCase(lambda: None)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream, result_decorators=[LoggingDecorator])
        self.run_test_runner(runner, test)
        self.assertLength(1, calls)

    def test_stopTestRun(self):
        """run should call result.stopTestRun()"""
        calls = []

        class LoggingDecorator(ExtendedToOriginalDecorator):

            def stopTestRun(self):
                ExtendedToOriginalDecorator.stopTestRun(self)
                calls.append('stopTestRun')
        test = unittest.FunctionTestCase(lambda: None)
        stream = StringIO()
        runner = tests.TextTestRunner(stream=stream, result_decorators=[LoggingDecorator])
        self.run_test_runner(runner, test)
        self.assertLength(1, calls)

    def test_unicode_test_output_on_ascii_stream(self):
        """Showing results should always succeed even on an ascii console"""

        class FailureWithUnicode(tests.TestCase):

            def test_log_unicode(self):
                self.log('☆')
                self.fail('Now print that log!')
        bio = BytesIO()
        out = TextIOWrapper(bio, 'ascii', 'backslashreplace')
        self.overrideAttr(osutils, 'get_terminal_encoding', lambda trace=False: 'ascii')
        self.run_test_runner(tests.TextTestRunner(stream=out), FailureWithUnicode('test_log_unicode'))
        out.flush()
        self.assertContainsRe(bio.getvalue(), b'(?:Text attachment: )?log(?:\n-+\n|: {{{)\\d+\\.\\d+  \\\\u2606(?:\n-+\n|}}}\n)')