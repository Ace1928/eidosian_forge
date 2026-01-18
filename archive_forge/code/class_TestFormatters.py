import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestFormatters(unittest.TestCase):

    def test_filename(self):
        completer = autocomplete.FilenameCompletion()
        last_part_of_filename = completer.format
        self.assertEqual(last_part_of_filename('abc'), 'abc')
        self.assertEqual(last_part_of_filename('abc/'), 'abc/')
        self.assertEqual(last_part_of_filename('abc/efg'), 'efg')
        self.assertEqual(last_part_of_filename('abc/efg/'), 'efg/')
        self.assertEqual(last_part_of_filename('/abc'), 'abc')
        self.assertEqual(last_part_of_filename('ab.c/e.f.g/'), 'e.f.g/')

    def test_attribute(self):
        self.assertEqual(autocomplete._after_last_dot('abc.edf'), 'edf')