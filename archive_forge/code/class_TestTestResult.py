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
class TestTestResult(tests.TestCase):

    def check_timing(self, test_case, expected_re):
        result = tests.TextTestResult(StringIO(), descriptions=0, verbosity=1)
        capture = testtools.testresult.doubles.ExtendedTestResult()
        test_case.run(MultiTestResult(result, capture))
        run_case = capture._events[0][1]
        timed_string = result._testTimeString(run_case)
        self.assertContainsRe(timed_string, expected_re)

    def test_test_reporting(self):

        class ShortDelayTestCase(tests.TestCase):

            def test_short_delay(self):
                time.sleep(0.003)

            def test_short_benchmark(self):
                self.time(time.sleep, 0.003)
        self.check_timing(ShortDelayTestCase('test_short_delay'), '^ +[0-9]+ms$')
        self.check_timing(ShortDelayTestCase('test_short_benchmark'), '^ +[0-9]+ms\\*$')

    def test_unittest_reporting_unittest_class(self):

        class ShortDelayTestCase(unittest.TestCase):

            def test_short_delay(self):
                time.sleep(0.003)
        self.check_timing(ShortDelayTestCase('test_short_delay'), '^ +[0-9]+ms$')

    def _time_hello_world_encoding(self):
        """Profile two sleep calls

        This is used to exercise the test framework.
        """
        self.time(str, b'hello', errors='replace')
        self.time(str, b'world', errors='replace')

    def test_lsprofiling(self):
        """Verbose test result prints lsprof statistics from test cases."""
        self.requireFeature(features.lsprof_feature)
        result_stream = StringIO()
        result = breezy.tests.VerboseTestResult(result_stream, descriptions=0, verbosity=2)
        example_test_case = TestTestResult('_time_hello_world_encoding')
        example_test_case._gather_lsprof_in_benchmarks = True
        example_test_case.run(result)
        output = result_stream.getvalue()
        self.assertContainsRe(output, "LSProf output for <class 'str'>\\(\\(b'hello',\\), {'errors': 'replace'}\\)")
        self.assertContainsRe(output, "LSProf output for <class 'str'>\\(\\(b'world',\\), {'errors': 'replace'}\\)")
        self.assertContainsRe(output, ' *CallCount *Recursive *Total\\(ms\\) *Inline\\(ms\\) *module:lineno\\(function\\)\\n')
        self.assertContainsRe(output, "( +1 +0 +0\\.\\d+ +0\\.\\d+ +<method 'disable' of '_lsprof\\.Profiler' objects>\\n)?")

    def test_uses_time_from_testtools(self):
        """Test case timings in verbose results should use testtools times"""
        import datetime

        class TimeAddedVerboseTestResult(tests.VerboseTestResult):

            def startTest(self, test):
                self.time(datetime.datetime.utcfromtimestamp(1.145))
                super().startTest(test)

            def addSuccess(self, test):
                self.time(datetime.datetime.utcfromtimestamp(51.147))
                super().addSuccess(test)

            def report_tests_starting(self):
                pass
        sio = StringIO()
        self.get_passing_test().run(TimeAddedVerboseTestResult(sio, 0, 2))
        self.assertEndsWith(sio.getvalue(), 'OK    50002ms\n')

    def test_known_failure(self):
        """Using knownFailure should trigger several result actions."""

        class InstrumentedTestResult(tests.ExtendedTestResult):

            def stopTestRun(self):
                pass

            def report_tests_starting(self):
                pass

            def report_known_failure(self, test, err=None, details=None):
                self._call = (test, 'known failure')
        result = InstrumentedTestResult(None, None, None, None)

        class Test(tests.TestCase):

            def test_function(self):
                self.knownFailure('failed!')
        test = Test('test_function')
        test.run(result)
        self.assertEqual(2, len(result._call))
        self.assertEqual(test.id(), result._call[0].id())
        self.assertEqual('known failure', result._call[1])
        self.assertEqual(1, result.known_failure_count)
        self.assertTrue(result.wasSuccessful())

    def test_verbose_report_known_failure(self):
        result_stream = StringIO()
        result = breezy.tests.VerboseTestResult(result_stream, descriptions=0, verbosity=2)
        _get_test('test_xfail').run(result)
        self.assertContainsRe(result_stream.getvalue(), '\n\\S+\\.test_xfail\\s+XFAIL\\s+\\d+ms\n\\s*(?:Text attachment: )?reason(?:\n-+\n|: {{{)this_fails(?:\n-+\n|}}}\n)')

    def get_passing_test(self):
        """Return a test object that can't be run usefully."""

        def passing_test():
            pass
        return unittest.FunctionTestCase(passing_test)

    def test_add_not_supported(self):
        """Test the behaviour of invoking addNotSupported."""

        class InstrumentedTestResult(tests.ExtendedTestResult):

            def stopTestRun(self):
                pass

            def report_tests_starting(self):
                pass

            def report_unsupported(self, test, feature):
                self._call = (test, feature)
        result = InstrumentedTestResult(None, None, None, None)
        test = SampleTestCase('_test_pass')
        feature = features.Feature()
        result.startTest(test)
        result.addNotSupported(test, feature)
        self.assertEqual(2, len(result._call))
        self.assertEqual(test, result._call[0])
        self.assertEqual(feature, result._call[1])
        self.assertTrue(result.wasSuccessful())
        self.assertEqual(1, result.unsupported['Feature'])
        result.addNotSupported(test, feature)
        self.assertEqual(2, result.unsupported['Feature'])

    def test_verbose_report_unsupported(self):
        result_stream = StringIO()
        result = breezy.tests.VerboseTestResult(result_stream, descriptions=0, verbosity=2)
        test = self.get_passing_test()
        feature = features.Feature()
        result.startTest(test)
        prefix = len(result_stream.getvalue())
        result.report_unsupported(test, feature)
        output = result_stream.getvalue()[prefix:]
        lines = output.splitlines()
        self.assertStartsWith(lines[0], 'NODEP')
        self.assertEqual(lines[1], "    The feature 'Feature' is not available.")

    def test_unavailable_exception(self):
        """An UnavailableFeature being raised should invoke addNotSupported."""

        class InstrumentedTestResult(tests.ExtendedTestResult):

            def stopTestRun(self):
                pass

            def report_tests_starting(self):
                pass

            def addNotSupported(self, test, feature):
                self._call = (test, feature)
        result = InstrumentedTestResult(None, None, None, None)
        feature = features.Feature()

        class Test(tests.TestCase):

            def test_function(self):
                raise tests.UnavailableFeature(feature)
        test = Test('test_function')
        test.run(result)
        self.assertEqual(2, len(result._call))
        self.assertEqual(test.id(), result._call[0].id())
        self.assertEqual(feature, result._call[1])
        self.assertEqual(0, result.error_count)

    def test_strict_with_unsupported_feature(self):
        result = tests.TextTestResult(StringIO(), descriptions=0, verbosity=1)
        test = self.get_passing_test()
        feature = 'Unsupported Feature'
        result.addNotSupported(test, feature)
        self.assertFalse(result.wasStrictlySuccessful())
        self.assertEqual(None, result._extractBenchmarkTime(test))

    def test_strict_with_known_failure(self):
        result = tests.TextTestResult(StringIO(), descriptions=0, verbosity=1)
        test = _get_test('test_xfail')
        test.run(result)
        self.assertFalse(result.wasStrictlySuccessful())
        self.assertEqual(None, result._extractBenchmarkTime(test))

    def test_strict_with_success(self):
        result = tests.TextTestResult(StringIO(), descriptions=0, verbosity=1)
        test = self.get_passing_test()
        result.addSuccess(test)
        self.assertTrue(result.wasStrictlySuccessful())
        self.assertEqual(None, result._extractBenchmarkTime(test))

    def test_startTests(self):
        """Starting the first test should trigger startTests."""

        class InstrumentedTestResult(tests.ExtendedTestResult):
            calls = 0

            def startTests(self):
                self.calls += 1
        result = InstrumentedTestResult(None, None, None, None)

        def test_function():
            pass
        test = unittest.FunctionTestCase(test_function)
        test.run(result)
        self.assertEqual(1, result.calls)

    def test_startTests_only_once(self):
        """With multiple tests startTests should still only be called once"""

        class InstrumentedTestResult(tests.ExtendedTestResult):
            calls = 0

            def startTests(self):
                self.calls += 1
        result = InstrumentedTestResult(None, None, None, None)
        suite = unittest.TestSuite([unittest.FunctionTestCase(lambda: None), unittest.FunctionTestCase(lambda: None)])
        suite.run(result)
        self.assertEqual(1, result.calls)
        self.assertEqual(2, result.count)