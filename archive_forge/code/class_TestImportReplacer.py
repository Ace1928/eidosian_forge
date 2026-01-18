import linecache
import os
import re
import sys
from .. import errors, lazy_import, osutils
from . import TestCase, TestCaseInTempDir
import %(root_name)s.%(sub_name)s.%(submoda_name)s as submoda7
class TestImportReplacer(ImportReplacerHelper):

    def test_import_root(self):
        """Test 'import root-XXX as root1'"""
        try:
            root1
        except NameError:
            pass
        else:
            self.fail('root1 was not supposed to exist yet')
        InstrumentedImportReplacer(scope=globals(), name='root1', module_path=[self.root_name], member=None, children={})
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root1, '__class__'))
        self.assertEqual(1, root1.var1)
        self.assertEqual('x', root1.func1('x'))
        self.assertEqual([('__getattribute__', 'var1'), ('_import', 'root1'), ('import', self.root_name, [], 0)], self.actions)

    def test_import_mod(self):
        """Test 'import root-XXX.mod-XXX as mod2'"""
        try:
            mod1
        except NameError:
            pass
        else:
            self.fail('mod1 was not supposed to exist yet')
        mod_path = self.root_name + '.' + self.mod_name
        InstrumentedImportReplacer(scope=globals(), name='mod1', module_path=[self.root_name, self.mod_name], member=None, children={})
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(mod1, '__class__'))
        self.assertEqual(2, mod1.var2)
        self.assertEqual('y', mod1.func2('y'))
        self.assertEqual([('__getattribute__', 'var2'), ('_import', 'mod1'), ('import', mod_path, [], 0)], self.actions)

    def test_import_mod_from_root(self):
        """Test 'from root-XXX import mod-XXX as mod2'"""
        try:
            mod2
        except NameError:
            pass
        else:
            self.fail('mod2 was not supposed to exist yet')
        InstrumentedImportReplacer(scope=globals(), name='mod2', module_path=[self.root_name], member=self.mod_name, children={})
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(mod2, '__class__'))
        self.assertEqual(2, mod2.var2)
        self.assertEqual('y', mod2.func2('y'))
        self.assertEqual([('__getattribute__', 'var2'), ('_import', 'mod2'), ('import', self.root_name, [self.mod_name], 0)], self.actions)

    def test_import_root_and_mod(self):
        """Test 'import root-XXX.mod-XXX' remapping both to root3.mod3"""
        try:
            root3
        except NameError:
            pass
        else:
            self.fail('root3 was not supposed to exist yet')
        InstrumentedImportReplacer(scope=globals(), name='root3', module_path=[self.root_name], member=None, children={'mod3': ([self.root_name, self.mod_name], None, {})})
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root3, '__class__'))
        self.assertEqual(1, root3.var1)
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root3.mod3, '__class__'))
        self.assertEqual(2, root3.mod3.var2)
        mod_path = self.root_name + '.' + self.mod_name
        self.assertEqual([('__getattribute__', 'var1'), ('_import', 'root3'), ('import', self.root_name, [], 0), ('__getattribute__', 'var2'), ('_import', 'mod3'), ('import', mod_path, [], 0)], self.actions)

    def test_import_root_and_root_mod(self):
        """Test that 'import root, root.mod' can be done.

        The second import should re-use the first one, and just add
        children to be imported.
        """
        try:
            root4
        except NameError:
            pass
        else:
            self.fail('root4 was not supposed to exist yet')
        InstrumentedImportReplacer(scope=globals(), name='root4', module_path=[self.root_name], member=None, children={})
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root4, '__class__'))
        children = object.__getattribute__(root4, '_import_replacer_children')
        children['mod4'] = ([self.root_name, self.mod_name], None, {})
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root4.mod4, '__class__'))
        self.assertEqual(2, root4.mod4.var2)
        mod_path = self.root_name + '.' + self.mod_name
        self.assertEqual([('__getattribute__', 'mod4'), ('_import', 'root4'), ('import', self.root_name, [], 0), ('__getattribute__', 'var2'), ('_import', 'mod4'), ('import', mod_path, [], 0)], self.actions)

    def test_import_root_sub_submod(self):
        """Test import root.mod, root.sub.submoda, root.sub.submodb
        root should be a lazy import, with multiple children, who also
        have children to be imported.
        And when root is imported, the children should be lazy, and
        reuse the intermediate lazy object.
        """
        try:
            root5
        except NameError:
            pass
        else:
            self.fail('root5 was not supposed to exist yet')
        InstrumentedImportReplacer(scope=globals(), name='root5', module_path=[self.root_name], member=None, children={'mod5': ([self.root_name, self.mod_name], None, {}), 'sub5': ([self.root_name, self.sub_name], None, {'submoda5': ([self.root_name, self.sub_name, self.submoda_name], None, {}), 'submodb5': ([self.root_name, self.sub_name, self.submodb_name], None, {})})})
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5, '__class__'))
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5.mod5, '__class__'))
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5.sub5, '__class__'))
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5.sub5.submoda5, '__class__'))
        self.assertEqual(InstrumentedImportReplacer, object.__getattribute__(root5.sub5.submodb5, '__class__'))
        self.assertEqual(2, root5.mod5.var2)
        self.assertEqual(4, root5.sub5.submoda5.var4)
        self.assertEqual(5, root5.sub5.submodb5.var5)
        mod_path = self.root_name + '.' + self.mod_name
        sub_path = self.root_name + '.' + self.sub_name
        submoda_path = sub_path + '.' + self.submoda_name
        submodb_path = sub_path + '.' + self.submodb_name
        self.assertEqual([('__getattribute__', 'mod5'), ('_import', 'root5'), ('import', self.root_name, [], 0), ('__getattribute__', 'submoda5'), ('_import', 'sub5'), ('import', sub_path, [], 0), ('__getattribute__', 'var2'), ('_import', 'mod5'), ('import', mod_path, [], 0), ('__getattribute__', 'var4'), ('_import', 'submoda5'), ('import', submoda_path, [], 0), ('__getattribute__', 'var5'), ('_import', 'submodb5'), ('import', submodb_path, [], 0)], self.actions)