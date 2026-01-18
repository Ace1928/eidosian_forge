import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestGetCompleter(unittest.TestCase):

    def test_no_completers(self):
        self.assertTupleEqual(autocomplete.get_completer([], 0, ''), ([], None))

    def test_one_completer_without_matches_returns_empty_list_and_none(self):
        a = completer([])
        self.assertTupleEqual(autocomplete.get_completer([a], 0, ''), ([], None))

    def test_one_completer_returns_matches_and_completer(self):
        a = completer(['a'])
        self.assertTupleEqual(autocomplete.get_completer([a], 0, ''), (['a'], a))

    def test_two_completers_with_matches_returns_first_matches(self):
        a = completer(['a'])
        b = completer(['b'])
        self.assertEqual(autocomplete.get_completer([a, b], 0, ''), (['a'], a))

    def test_first_non_none_completer_matches_are_returned(self):
        a = completer([])
        b = completer(['a'])
        self.assertEqual(autocomplete.get_completer([a, b], 0, ''), ([], None))

    def test_only_completer_returns_None(self):
        a = completer(None)
        self.assertEqual(autocomplete.get_completer([a], 0, ''), ([], None))

    def test_first_completer_returns_None(self):
        a = completer(None)
        b = completer(['a'])
        self.assertEqual(autocomplete.get_completer([a, b], 0, ''), (['a'], b))