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
class TestTestSuite(tests.TestCase):

    def test__test_suite_testmod_names(self):
        test_list = tests._test_suite_testmod_names()
        self.assertSubset(['breezy.tests.blackbox', 'breezy.tests.per_transport', 'breezy.tests.test_selftest'], test_list)

    def test__test_suite_modules_to_doctest(self):
        test_list = tests._test_suite_modules_to_doctest()
        if __doc__ is None:
            self.assertEqual([], test_list)
            return
        self.assertSubset(['breezy.timestamp'], test_list)

    def test_test_suite(self):
        calls = []

        def testmod_names():
            calls.append('testmod_names')
            return ['breezy.tests.blackbox.test_branch', 'breezy.tests.per_transport', 'breezy.tests.test_selftest']
        self.overrideAttr(tests, '_test_suite_testmod_names', testmod_names)

        def doctests():
            calls.append('modules_to_doctest')
            if __doc__ is None:
                return []
            return ['breezy.timestamp']
        self.overrideAttr(tests, '_test_suite_modules_to_doctest', doctests)
        expected_test_list = ['breezy.tests.blackbox.test_branch.TestBranch.test_branch', 'breezy.tests.per_transport.TransportTests.test_abspath(LocalTransport,LocalURLServer)', 'breezy.tests.test_selftest.TestTestSuite.test_test_suite']
        suite = tests.test_suite()
        self.assertEqual({'testmod_names', 'modules_to_doctest'}, set(calls))
        self.assertSubset(expected_test_list, _test_ids(suite))

    def test_test_suite_list_and_start(self):
        test_list = ['breezy.tests.test_selftest.TestTestSuite.test_test_suite']
        suite = tests.test_suite(test_list, ['breezy.tests.test_selftest.TestTestSuite'])
        self.assertEqual(test_list, _test_ids(suite))