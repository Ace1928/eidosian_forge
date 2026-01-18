import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestImportProcessor(TestCase):
    """Test that ImportProcessor can turn import texts into lazy imports"""

    def check(self, expected, text):
        proc = lazy_import.ImportProcessor()
        proc._build_map(text)
        self.assertEqual(expected, proc.imports, 'Incorrect processing of:\n%s\n%s\n!=\n%s' % (text, expected, proc.imports))

    def test_import_one(self):
        exp = {'one': (['one'], None, {})}
        self.check(exp, 'import one')
        self.check(exp, '\nimport one\n')

    def test_import_one_two(self):
        exp = {'one': (['one'], None, {'two': (['one', 'two'], None, {})})}
        self.check(exp, 'import one.two')
        self.check(exp, 'import one, one.two')
        self.check(exp, 'import one\nimport one.two')

    def test_import_as(self):
        exp = {'two': (['one'], None, {})}
        self.check(exp, 'import one as two')

    def test_import_many(self):
        exp = {'one': (['one'], None, {'two': (['one', 'two'], None, {'three': (['one', 'two', 'three'], None, {})}), 'four': (['one', 'four'], None, {})}), 'five': (['one', 'five'], None, {})}
        self.check(exp, 'import one.two.three, one.four, one.five as five')
        self.check(exp, 'import one.five as five\nimport one\nimport one.two.three\nimport one.four\n')

    def test_from_one_import_two(self):
        exp = {'two': (['one'], 'two', {})}
        self.check(exp, 'from one import two\n')
        self.check(exp, 'from one import (\n    two,\n    )\n')

    def test_from_one_import_two_two(self):
        exp = {'two': (['one'], 'two', {})}
        self.check(exp, 'from one import two\n')
        self.check(exp, 'from one import (two)\n')
        self.check(exp, 'from one import (two,)\n')
        self.check(exp, 'from one import two as two\n')
        self.check(exp, 'from one import (\n    two,\n    )\n')

    def test_from_many(self):
        exp = {'two': (['one'], 'two', {}), 'three': (['one', 'two'], 'three', {}), 'five': (['one', 'two'], 'four', {})}
        self.check(exp, 'from one import two\nfrom one.two import three, four as five\n')
        self.check(exp, 'from one import two\nfrom one.two import (\n    three,\n    four as five,\n    )\n')

    def test_mixed(self):
        exp = {'two': (['one'], 'two', {}), 'three': (['one', 'two'], 'three', {}), 'five': (['one', 'two'], 'four', {}), 'one': (['one'], None, {'two': (['one', 'two'], None, {})})}
        self.check(exp, 'from one import two\nfrom one.two import three, four as five\nimport one.two')
        self.check(exp, 'from one import two\nfrom one.two import (\n    three,\n    four as five,\n    )\nimport one\nimport one.two\n')

    def test_incorrect_line(self):
        proc = lazy_import.ImportProcessor()
        self.assertRaises(lazy_import.InvalidImportLine, proc._build_map, 'foo bar baz')
        self.assertRaises(lazy_import.InvalidImportLine, proc._build_map, 'improt foo')
        self.assertRaises(lazy_import.InvalidImportLine, proc._build_map, 'importfoo')
        self.assertRaises(lazy_import.InvalidImportLine, proc._build_map, 'fromimport')

    def test_name_collision(self):
        proc = lazy_import.ImportProcessor()
        proc._build_map('import foo')
        self.assertRaises(lazy_import.ImportNameCollision, proc._build_map, 'import bar as foo')
        self.assertRaises(lazy_import.ImportNameCollision, proc._build_map, 'from foo import bar as foo')
        self.assertRaises(lazy_import.ImportNameCollision, proc._build_map, 'from bar import foo')

    def test_relative_imports(self):
        proc = lazy_import.ImportProcessor()
        self.assertRaises(ImportError, proc._build_map, 'import .bar as foo')
        self.assertRaises(ImportError, proc._build_map, 'from .foo import bar as foo')
        self.assertRaises(ImportError, proc._build_map, 'from .bar import foo')