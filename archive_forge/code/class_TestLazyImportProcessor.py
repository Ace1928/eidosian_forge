import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestLazyImportProcessor(ImportReplacerHelper):

    def test_root(self):
        try:
            root6
        except NameError:
            pass
        else:
            self.fail('root6 was not supposed to exist yet')
        text = 'import {} as root6'.format(self.root_name)
        proc = lazy_import.ImportProcessor(InstrumentedImportReplacer)
        proc.lazy_import(scope=globals(), text=text)
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root6, '__class__'))
        self.assertEqual(1, root6.var1)
        self.assertEqual('x', root6.func1('x'))
        self.assertEqual([('__getattribute__', 'var1'), ('_import', 'root6'), ('import', self.root_name, [], 0)], self.actions)

    def test_import_deep(self):
        """Test import root.mod, root.sub.submoda, root.sub.submodb
        root should be a lazy import, with multiple children, who also
        have children to be imported.
        And when root is imported, the children should be lazy, and
        reuse the intermediate lazy object.
        """
        try:
            submoda7
        except NameError:
            pass
        else:
            self.fail('submoda7 was not supposed to exist yet')
        text = 'import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7\n' % self.__dict__
        proc = lazy_import.ImportProcessor(InstrumentedImportReplacer)
        proc.lazy_import(scope=globals(), text=text)
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(submoda7, '__class__'))
        self.assertEqual(4, submoda7.var4)
        sub_path = self.root_name + '.' + self.sub_name
        submoda_path = sub_path + '.' + self.submoda_name
        self.assertEqual([('__getattribute__', 'var4'), ('_import', 'submoda7'), ('import', submoda_path, [], 0)], self.actions)

    def test_lazy_import(self):
        """Smoke test that lazy_import() does the right thing"""
        try:
            root8
        except NameError:
            pass
        else:
            self.fail('root8 was not supposed to exist yet')
        lazy_import.lazy_import(globals(), 'import {} as root8'.format(self.root_name), lazy_import_class=InstrumentedImportReplacer)
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root8, '__class__'))
        self.assertEqual(1, root8.var1)
        self.assertEqual(1, root8.var1)
        self.assertEqual(1, root8.func1(1))
        self.assertEqual([('__getattribute__', 'var1'), ('_import', 'root8'), ('import', self.root_name, [], 0)], self.actions)