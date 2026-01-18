import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestCumulativeCompleter(unittest.TestCase):

    def completer(self, matches):
        mock_completer = autocomplete.BaseCompletionType()
        mock_completer.matches = mock.Mock(return_value=matches)
        return mock_completer

    def test_no_completers_fails(self):
        with self.assertRaises(ValueError):
            autocomplete.CumulativeCompleter([])

    def test_one_empty_completer_returns_empty(self):
        a = self.completer([])
        cumulative = autocomplete.CumulativeCompleter([a])
        self.assertEqual(cumulative.matches(3, 'abc'), set())

    def test_one_none_completer_returns_none(self):
        a = self.completer(None)
        cumulative = autocomplete.CumulativeCompleter([a])
        self.assertEqual(cumulative.matches(3, 'abc'), None)

    def test_two_completers_get_both(self):
        a = self.completer(['a'])
        b = self.completer(['b'])
        cumulative = autocomplete.CumulativeCompleter([a, b])
        self.assertEqual(cumulative.matches(3, 'abc'), {'a', 'b'})