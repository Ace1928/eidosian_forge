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
class TestUncollectedWarnings(_Selftest, tests.TestCase):
    """Check a test case still alive after being run emits a warning"""

    class Test(tests.TestCase):

        def test_pass(self):
            pass

        def test_self_ref(self):
            self.also_self = self.test_self_ref

        def test_skip(self):
            self.skipTest("Don't need")

    def _get_suite(self):
        return TestUtil.TestSuite([self.Test('test_pass'), self.Test('test_self_ref'), self.Test('test_skip')])

    def _run_selftest_with_suite(self, **kwargs):
        old_flags = tests.selftest_debug_flags
        tests.selftest_debug_flags = old_flags.union(['uncollected_cases'])
        gc_on = gc.isenabled()
        if gc_on:
            gc.disable()
        try:
            output = self._run_selftest(test_suite_factory=self._get_suite, **kwargs)
        finally:
            if gc_on:
                gc.enable()
            tests.selftest_debug_flags = old_flags
        self.assertNotContainsRe(output, b'Uncollected test case.*test_pass')
        self.assertContainsRe(output, b'Uncollected test case.*test_self_ref')
        return output

    def test_testsuite(self):
        self._run_selftest_with_suite()

    def test_pattern(self):
        out = self._run_selftest_with_suite(pattern='test_(?:pass|self_ref)$')
        self.assertNotContainsRe(out, b'test_skip')

    def test_exclude_pattern(self):
        out = self._run_selftest_with_suite(exclude_pattern='test_skip$')
        self.assertNotContainsRe(out, b'test_skip')

    def test_random_seed(self):
        self._run_selftest_with_suite(random_seed='now')

    def test_matching_tests_first(self):
        self._run_selftest_with_suite(matching_tests_first=True, pattern='test_self_ref$')

    def test_starting_with_and_exclude(self):
        out = self._run_selftest_with_suite(starting_with=['bt.'], exclude_pattern='test_skip$')
        self.assertNotContainsRe(out, b'test_skip')

    def test_additonal_decorator(self):
        self._run_selftest_with_suite(suite_decorators=[tests.TestDecorator])