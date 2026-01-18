import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestDictKeyCompletion(unittest.TestCase):

    def test_set_of_keys_returned_when_matches_found(self):
        com = autocomplete.DictKeyCompletion()
        local = {'d': {'ab': 1, 'cd': 2}}
        self.assertSetEqual(com.matches(2, 'd[', locals_=local), {"'ab']", "'cd']"})

    def test_none_returned_when_eval_error(self):
        com = autocomplete.DictKeyCompletion()
        local = {'e': {'ab': 1, 'cd': 2}}
        self.assertEqual(com.matches(2, 'd[', locals_=local), None)

    def test_none_returned_when_not_dict_type(self):
        com = autocomplete.DictKeyCompletion()
        local = {'l': ['ab', 'cd']}
        self.assertEqual(com.matches(2, 'l[', locals_=local), None)

    def test_none_returned_when_no_matches_left(self):
        com = autocomplete.DictKeyCompletion()
        local = {'d': {'ab': 1, 'cd': 2}}
        self.assertEqual(com.matches(3, 'd[r', locals_=local), None)

    def test_obj_that_does_not_allow_conversion_to_bool(self):
        com = autocomplete.DictKeyCompletion()
        local = {'mNumPy': MockNumPy()}
        self.assertEqual(com.matches(7, 'mNumPy[', locals_=local), None)