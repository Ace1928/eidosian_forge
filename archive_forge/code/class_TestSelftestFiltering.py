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
class TestSelftestFiltering(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.suite = TestUtil.TestSuite()
        self.loader = TestUtil.TestLoader()
        self.suite.addTest(self.loader.loadTestsFromModule(sys.modules['breezy.tests.test_selftest']))
        self.all_names = _test_ids(self.suite)

    def test_condition_id_re(self):
        test_name = 'breezy.tests.test_selftest.TestSelftestFiltering.test_condition_id_re'
        filtered_suite = tests.filter_suite_by_condition(self.suite, tests.condition_id_re('test_condition_id_re'))
        self.assertEqual([test_name], _test_ids(filtered_suite))

    def test_condition_id_in_list(self):
        test_names = ['breezy.tests.test_selftest.TestSelftestFiltering.test_condition_id_in_list']
        id_list = tests.TestIdList(test_names)
        filtered_suite = tests.filter_suite_by_condition(self.suite, tests.condition_id_in_list(id_list))
        my_pattern = 'TestSelftestFiltering.*test_condition_id_in_list'
        re_filtered = tests.filter_suite_by_re(self.suite, my_pattern)
        self.assertEqual(_test_ids(re_filtered), _test_ids(filtered_suite))

    def test_condition_id_startswith(self):
        klass = 'breezy.tests.test_selftest.TestSelftestFiltering.'
        start1 = klass + 'test_condition_id_starts'
        start2 = klass + 'test_condition_id_in'
        test_names = [klass + 'test_condition_id_in_list', klass + 'test_condition_id_startswith']
        filtered_suite = tests.filter_suite_by_condition(self.suite, tests.condition_id_startswith([start1, start2]))
        self.assertEqual(test_names, _test_ids(filtered_suite))

    def test_condition_isinstance(self):
        filtered_suite = tests.filter_suite_by_condition(self.suite, tests.condition_isinstance(self.__class__))
        class_pattern = 'breezy.tests.test_selftest.TestSelftestFiltering.'
        re_filtered = tests.filter_suite_by_re(self.suite, class_pattern)
        self.assertEqual(_test_ids(re_filtered), _test_ids(filtered_suite))

    def test_exclude_tests_by_condition(self):
        excluded_name = 'breezy.tests.test_selftest.TestSelftestFiltering.test_exclude_tests_by_condition'
        filtered_suite = tests.exclude_tests_by_condition(self.suite, lambda x: x.id() == excluded_name)
        self.assertEqual(len(self.all_names) - 1, filtered_suite.countTestCases())
        self.assertFalse(excluded_name in _test_ids(filtered_suite))
        remaining_names = list(self.all_names)
        remaining_names.remove(excluded_name)
        self.assertEqual(remaining_names, _test_ids(filtered_suite))

    def test_exclude_tests_by_re(self):
        self.all_names = _test_ids(self.suite)
        filtered_suite = tests.exclude_tests_by_re(self.suite, 'exclude_tests_by_re')
        excluded_name = 'breezy.tests.test_selftest.TestSelftestFiltering.test_exclude_tests_by_re'
        self.assertEqual(len(self.all_names) - 1, filtered_suite.countTestCases())
        self.assertFalse(excluded_name in _test_ids(filtered_suite))
        remaining_names = list(self.all_names)
        remaining_names.remove(excluded_name)
        self.assertEqual(remaining_names, _test_ids(filtered_suite))

    def test_filter_suite_by_condition(self):
        test_name = 'breezy.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_condition'
        filtered_suite = tests.filter_suite_by_condition(self.suite, lambda x: x.id() == test_name)
        self.assertEqual([test_name], _test_ids(filtered_suite))

    def test_filter_suite_by_re(self):
        filtered_suite = tests.filter_suite_by_re(self.suite, 'test_filter_suite_by_r')
        filtered_names = _test_ids(filtered_suite)
        self.assertEqual(filtered_names, ['breezy.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_re'])

    def test_filter_suite_by_id_list(self):
        test_list = ['breezy.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_id_list']
        filtered_suite = tests.filter_suite_by_id_list(self.suite, tests.TestIdList(test_list))
        filtered_names = _test_ids(filtered_suite)
        self.assertEqual(filtered_names, ['breezy.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_id_list'])

    def test_filter_suite_by_id_startswith(self):
        klass = 'breezy.tests.test_selftest.TestSelftestFiltering.'
        start1 = klass + 'test_filter_suite_by_id_starts'
        start2 = klass + 'test_filter_suite_by_id_li'
        test_list = [klass + 'test_filter_suite_by_id_list', klass + 'test_filter_suite_by_id_startswith']
        filtered_suite = tests.filter_suite_by_id_startswith(self.suite, [start1, start2])
        self.assertEqual(test_list, _test_ids(filtered_suite))

    def test_preserve_input(self):
        self.assertIs(self.suite, tests.preserve_input(self.suite))
        self.assertEqual('@#$', tests.preserve_input('@#$'))

    def test_randomize_suite(self):
        randomized_suite = tests.randomize_suite(self.suite)
        self.assertEqual(set(_test_ids(self.suite)), set(_test_ids(randomized_suite)))
        self.assertNotEqual(self.all_names, _test_ids(randomized_suite))
        self.assertEqual(len(self.all_names), len(_test_ids(randomized_suite)))

    def test_split_suit_by_condition(self):
        self.all_names = _test_ids(self.suite)
        condition = tests.condition_id_re('test_filter_suite_by_r')
        split_suite = tests.split_suite_by_condition(self.suite, condition)
        filtered_name = 'breezy.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_re'
        self.assertEqual([filtered_name], _test_ids(split_suite[0]))
        self.assertFalse(filtered_name in _test_ids(split_suite[1]))
        remaining_names = list(self.all_names)
        remaining_names.remove(filtered_name)
        self.assertEqual(remaining_names, _test_ids(split_suite[1]))

    def test_split_suit_by_re(self):
        self.all_names = _test_ids(self.suite)
        split_suite = tests.split_suite_by_re(self.suite, 'test_filter_suite_by_r')
        filtered_name = 'breezy.tests.test_selftest.TestSelftestFiltering.test_filter_suite_by_re'
        self.assertEqual([filtered_name], _test_ids(split_suite[0]))
        self.assertFalse(filtered_name in _test_ids(split_suite[1]))
        remaining_names = list(self.all_names)
        remaining_names.remove(filtered_name)
        self.assertEqual(remaining_names, _test_ids(split_suite[1]))