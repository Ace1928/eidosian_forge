import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestFromToMap(TestCase):
    """Directly test the conversion of 'from foo import bar' syntax"""

    def check_result(self, expected, from_strings):
        proc = lazy_import.ImportProcessor()
        for from_str in from_strings:
            proc._convert_from_str(from_str)
        self.assertEqual(expected, proc.imports, 'Import of %r was not converted correctly %s != %s' % (from_strings, expected, proc.imports))

    def test_from_one_import_two(self):
        self.check_result({'two': (['one'], 'two', {})}, ['from one import two'])

    def test_from_one_import_two_as_three(self):
        self.check_result({'three': (['one'], 'two', {})}, ['from one import two as three'])

    def test_from_one_import_two_three(self):
        two_three_map = {'two': (['one'], 'two', {}), 'three': (['one'], 'three', {})}
        self.check_result(two_three_map, ['from one import two, three'])
        self.check_result(two_three_map, ['from one import two', 'from one import three'])

    def test_from_one_two_import_three(self):
        self.check_result({'three': (['one', 'two'], 'three', {})}, ['from one.two import three'])