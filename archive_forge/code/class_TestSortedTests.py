import doctest
from pprint import pformat
import unittest
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import DocTestMatches, Equals
from testtools.testresult.doubles import StreamResult as LoggingStream
from testtools.testsuite import FixtureSuite, sorted_tests
from testtools.tests.helpers import LoggingResult
class TestSortedTests(TestCase):

    def test_sorts_custom_suites(self):
        a = PlaceHolder('a')
        b = PlaceHolder('b')

        class Subclass(unittest.TestSuite):

            def sort_tests(self):
                self._tests = sorted_tests(self, True)
        input_suite = Subclass([b, a])
        suite = sorted_tests(input_suite)
        self.assertEqual([a, b], list(iterate_tests(suite)))
        self.assertEqual([input_suite], list(iter(suite)))

    def test_custom_suite_without_sort_tests_works(self):
        a = PlaceHolder('a')
        b = PlaceHolder('b')

        class Subclass(unittest.TestSuite):
            pass
        input_suite = Subclass([b, a])
        suite = sorted_tests(input_suite)
        self.assertEqual([b, a], list(iterate_tests(suite)))
        self.assertEqual([input_suite], list(iter(suite)))

    def test_sorts_simple_suites(self):
        a = PlaceHolder('a')
        b = PlaceHolder('b')
        suite = sorted_tests(unittest.TestSuite([b, a]))
        self.assertEqual([a, b], list(iterate_tests(suite)))

    def test_duplicate_simple_suites(self):
        a = PlaceHolder('a')
        b = PlaceHolder('b')
        c = PlaceHolder('a')
        self.assertRaises(ValueError, sorted_tests, unittest.TestSuite([a, b, c]))

    def test_multiple_duplicates(self):
        a = PlaceHolder('a')
        b = PlaceHolder('b')
        c = PlaceHolder('a')
        d = PlaceHolder('b')
        error = self.assertRaises(ValueError, sorted_tests, unittest.TestSuite([a, b, c, d]))
        self.assertThat(str(error), Equals('Duplicate test ids detected: {}'.format(pformat({'a': 2, 'b': 2}))))