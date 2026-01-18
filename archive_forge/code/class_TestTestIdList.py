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
class TestTestIdList(tests.TestCase):

    def _create_id_list(self, test_list):
        return tests.TestIdList(test_list)

    def _create_suite(self, test_id_list):

        class Stub(tests.TestCase):

            def test_foo(self):
                pass

        def _create_test_id(id):
            return lambda: id
        suite = TestUtil.TestSuite()
        for id in test_id_list:
            t = Stub('test_foo')
            t.id = _create_test_id(id)
            suite.addTest(t)
        return suite

    def _test_ids(self, test_suite):
        """Get the ids for the tests in a test suite."""
        return [t.id() for t in tests.iter_suite_tests(test_suite)]

    def test_empty_list(self):
        id_list = self._create_id_list([])
        self.assertEqual({}, id_list.tests)
        self.assertEqual({}, id_list.modules)

    def test_valid_list(self):
        id_list = self._create_id_list(['mod1.cl1.meth1', 'mod1.cl1.meth2', 'mod1.func1', 'mod1.cl2.meth2', 'mod1.submod1', 'mod1.submod2.cl1.meth1', 'mod1.submod2.cl2.meth2'])
        self.assertTrue(id_list.refers_to('mod1'))
        self.assertTrue(id_list.refers_to('mod1.submod1'))
        self.assertTrue(id_list.refers_to('mod1.submod2'))
        self.assertTrue(id_list.includes('mod1.cl1.meth1'))
        self.assertTrue(id_list.includes('mod1.submod1'))
        self.assertTrue(id_list.includes('mod1.func1'))

    def test_bad_chars_in_params(self):
        id_list = self._create_id_list(['mod1.cl1.meth1(xx.yy)'])
        self.assertTrue(id_list.refers_to('mod1'))
        self.assertTrue(id_list.includes('mod1.cl1.meth1(xx.yy)'))

    def test_module_used(self):
        id_list = self._create_id_list(['mod.class.meth'])
        self.assertTrue(id_list.refers_to('mod'))
        self.assertTrue(id_list.refers_to('mod.class'))
        self.assertTrue(id_list.refers_to('mod.class.meth'))

    def test_test_suite_matches_id_list_with_unknown(self):
        loader = TestUtil.TestLoader()
        suite = loader.loadTestsFromModuleName('breezy.tests.test_sampler')
        test_list = ['breezy.tests.test_sampler.DemoTest.test_nothing', 'bogus']
        not_found, duplicates = tests.suite_matches_id_list(suite, test_list)
        self.assertEqual(['bogus'], not_found)
        self.assertEqual([], duplicates)

    def test_suite_matches_id_list_with_duplicates(self):
        loader = TestUtil.TestLoader()
        suite = loader.loadTestsFromModuleName('breezy.tests.test_sampler')
        dupes = loader.suiteClass()
        for test in tests.iter_suite_tests(suite):
            dupes.addTest(test)
            dupes.addTest(test)
        test_list = ['breezy.tests.test_sampler.DemoTest.test_nothing']
        not_found, duplicates = tests.suite_matches_id_list(dupes, test_list)
        self.assertEqual([], not_found)
        self.assertEqual(['breezy.tests.test_sampler.DemoTest.test_nothing'], duplicates)