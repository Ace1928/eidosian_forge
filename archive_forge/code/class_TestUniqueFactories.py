from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
class TestUniqueFactories(TestCase):
    """Tests for getUniqueString, getUniqueInteger, unique_text_generator."""
    run_test_with = FullStackRunTest

    def test_getUniqueInteger(self):
        one = self.getUniqueInteger()
        self.assertEqual(1, one)
        two = self.getUniqueInteger()
        self.assertEqual(2, two)

    def test_getUniqueString(self):
        name_one = self.getUniqueString()
        self.assertEqual('%s-%d' % (self.id(), 1), name_one)
        name_two = self.getUniqueString()
        self.assertEqual('%s-%d' % (self.id(), 2), name_two)

    def test_getUniqueString_prefix(self):
        name_one = self.getUniqueString('foo')
        self.assertThat(name_one, Equals('foo-1'))
        name_two = self.getUniqueString('bar')
        self.assertThat(name_two, Equals('bar-2'))

    def test_unique_text_generator(self):
        prefix = self.getUniqueString()
        unique_text_generator = testcase.unique_text_generator(prefix)
        first_result = next(unique_text_generator)
        self.assertEqual('{}-{}'.format(prefix, 'Ḁ'), first_result)
        second_result = next(unique_text_generator)
        self.assertEqual('{}-{}'.format(prefix, 'ḁ'), second_result)

    def test_mods(self):
        self.assertEqual([0], list(testcase._mods(0, 5)))
        self.assertEqual([1], list(testcase._mods(1, 5)))
        self.assertEqual([2], list(testcase._mods(2, 5)))
        self.assertEqual([3], list(testcase._mods(3, 5)))
        self.assertEqual([4], list(testcase._mods(4, 5)))
        self.assertEqual([0, 1], list(testcase._mods(5, 5)))
        self.assertEqual([1, 1], list(testcase._mods(6, 5)))
        self.assertEqual([2, 1], list(testcase._mods(7, 5)))
        self.assertEqual([0, 2], list(testcase._mods(10, 5)))
        self.assertEqual([0, 0, 1], list(testcase._mods(25, 5)))
        self.assertEqual([1, 0, 1], list(testcase._mods(26, 5)))
        self.assertEqual([1], list(testcase._mods(1, 100)))
        self.assertEqual([0, 1], list(testcase._mods(100, 100)))
        self.assertEqual([0, 10], list(testcase._mods(1000, 100)))

    def test_unique_text(self):
        self.assertEqual('Ḁ', testcase._unique_text(base_cp=7680, cp_range=5, index=0))
        self.assertEqual('ḁ', testcase._unique_text(base_cp=7680, cp_range=5, index=1))
        self.assertEqual('Ḁḁ', testcase._unique_text(base_cp=7680, cp_range=5, index=5))
        self.assertEqual('ḃḂḁ', testcase._unique_text(base_cp=7680, cp_range=5, index=38))