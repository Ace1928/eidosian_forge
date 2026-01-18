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
class TestSelftest(tests.TestCase, SelfTestHelper):
    """Tests of breezy.tests.selftest."""

    def test_selftest_benchmark_parameter_invokes_test_suite__benchmark__(self):
        factory_called = []

        def factory():
            factory_called.append(True)
            return TestUtil.TestSuite()
        out = StringIO()
        err = StringIO()
        self.apply_redirected(out, err, None, breezy.tests.selftest, test_suite_factory=factory)
        self.assertEqual([True], factory_called)

    def factory(self):
        """A test suite factory."""

        class Test(tests.TestCase):

            def id(self):
                return __name__ + '.Test.' + self._testMethodName

            def a(self):
                pass

            def b(self):
                pass

            def c(telf):
                pass
        return TestUtil.TestSuite([Test('a'), Test('b'), Test('c')])

    def test_list_only(self):
        output = self.run_selftest(test_suite_factory=self.factory, list_only=True)
        self.assertEqual(3, len(output.readlines()))

    def test_list_only_filtered(self):
        output = self.run_selftest(test_suite_factory=self.factory, list_only=True, pattern='Test.b')
        self.assertEndsWith(output.getvalue(), b'Test.b\n')
        self.assertLength(1, output.readlines())

    def test_list_only_excludes(self):
        output = self.run_selftest(test_suite_factory=self.factory, list_only=True, exclude_pattern='Test.b')
        self.assertNotContainsRe(b'Test.b', output.getvalue())
        self.assertLength(2, output.readlines())

    def test_lsprof_tests(self):
        self.requireFeature(features.lsprof_feature)
        results = []

        class Test:

            def __call__(test, result):
                test.run(result)

            def run(test, result):
                results.append(result)

            def countTestCases(self):
                return 1
        self.run_selftest(test_suite_factory=Test, lsprof_tests=True)
        self.assertLength(1, results)
        self.assertIsInstance(results.pop(), ExtendedToOriginalDecorator)

    def test_random(self):
        output_123 = self.run_selftest(test_suite_factory=self.factory, list_only=True, random_seed='123')
        output_234 = self.run_selftest(test_suite_factory=self.factory, list_only=True, random_seed='234')
        self.assertNotEqual(output_123, output_234)
        self.assertLength(5, output_123.readlines())
        self.assertLength(5, output_234.readlines())

    def test_random_reuse_is_same_order(self):
        expected = self.run_selftest(test_suite_factory=self.factory, list_only=True, random_seed='123')
        repeated = self.run_selftest(test_suite_factory=self.factory, list_only=True, random_seed='123')
        self.assertEqual(expected.getvalue(), repeated.getvalue())

    def test_runner_class(self):
        self.requireFeature(features.subunit)
        from subunit import ProtocolTestCase
        stream = self.run_selftest(runner_class=tests.SubUnitBzrRunnerv1, test_suite_factory=self.factory)
        test = ProtocolTestCase(stream)
        result = unittest.TestResult()
        test.run(result)
        self.assertEqual(3, result.testsRun)

    def test_starting_with_single_argument(self):
        output = self.run_selftest(test_suite_factory=self.factory, starting_with=['breezy.tests.test_selftest.Test.a'], list_only=True)
        self.assertEqual(b'breezy.tests.test_selftest.Test.a\n', output.getvalue())

    def test_starting_with_multiple_argument(self):
        output = self.run_selftest(test_suite_factory=self.factory, starting_with=['breezy.tests.test_selftest.Test.a', 'breezy.tests.test_selftest.Test.b'], list_only=True)
        self.assertEqual(b'breezy.tests.test_selftest.Test.a\nbreezy.tests.test_selftest.Test.b\n', output.getvalue())

    def check_transport_set(self, transport_server):
        captured_transport = []

        def seen_transport(a_transport):
            captured_transport.append(a_transport)

        class Capture(tests.TestCase):

            def a(self):
                seen_transport(breezy.tests.default_transport)

        def factory():
            return TestUtil.TestSuite([Capture('a')])
        self.run_selftest(transport=transport_server, test_suite_factory=factory)
        self.assertEqual(transport_server, captured_transport[0])

    def test_transport_sftp(self):
        self.requireFeature(features.paramiko)
        from breezy.tests import stub_sftp
        self.check_transport_set(stub_sftp.SFTPAbsoluteServer)

    def test_transport_memory(self):
        self.check_transport_set(memory.MemoryServer)