import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestImportReplacerHelper(ImportReplacerHelper):

    def test_basic_import(self):
        """Test that a real import of these modules works"""
        sub_mod_path = '.'.join([self.root_name, self.sub_name, self.submoda_name])
        root = lazy_import._builtin_import(sub_mod_path, {}, {}, [], 0)
        self.assertEqual(1, root.var1)
        self.assertEqual(3, getattr(root, self.sub_name).var3)
        self.assertEqual(4, getattr(getattr(root, self.sub_name), self.submoda_name).var4)
        mod_path = '.'.join([self.root_name, self.mod_name])
        root = lazy_import._builtin_import(mod_path, {}, {}, [], 0)
        self.assertEqual(2, getattr(root, self.mod_name).var2)
        self.assertEqual([('import', sub_mod_path, [], 0), ('import', mod_path, [], 0)], self.actions)