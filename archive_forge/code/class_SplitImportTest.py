from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import traceback
import unittest
import pasta
from pasta.augment import import_utils
from pasta.base import ast_utils
from pasta.base import test_utils
from pasta.base import scope
import a
import b
from my_module import a
import b
from my_module import c
from my_module import a, b
from my_module import a as a_mod, b as unused_b_mod
import c as c_mod, d as unused_d_mod
import a
import b
import c
import b
import d
import a, b
import b, c
import d, a, e, f
import a, b, c
import b, c
import a.b
from a import b
import a
import a as ax
import a as ax2
import a as ax
class SplitImportTest(test_utils.TestCase):

    def test_split_normal_import(self):
        src = 'import aaa, bbb, ccc\n'
        t = ast.parse(src)
        import_node = t.body[0]
        sc = scope.analyze(t)
        import_utils.split_import(sc, import_node, import_node.names[1])
        self.assertEqual(2, len(t.body))
        self.assertEqual(ast.Import, type(t.body[1]))
        self.assertEqual([alias.name for alias in t.body[0].names], ['aaa', 'ccc'])
        self.assertEqual([alias.name for alias in t.body[1].names], ['bbb'])

    def test_split_from_import(self):
        src = 'from aaa import bbb, ccc, ddd\n'
        t = ast.parse(src)
        import_node = t.body[0]
        sc = scope.analyze(t)
        import_utils.split_import(sc, import_node, import_node.names[1])
        self.assertEqual(2, len(t.body))
        self.assertEqual(ast.ImportFrom, type(t.body[1]))
        self.assertEqual(t.body[0].module, 'aaa')
        self.assertEqual(t.body[1].module, 'aaa')
        self.assertEqual([alias.name for alias in t.body[0].names], ['bbb', 'ddd'])

    def test_split_imports_with_alias(self):
        src = 'import aaa as a, bbb as b, ccc as c\n'
        t = ast.parse(src)
        import_node = t.body[0]
        sc = scope.analyze(t)
        import_utils.split_import(sc, import_node, import_node.names[1])
        self.assertEqual(2, len(t.body))
        self.assertEqual([alias.name for alias in t.body[0].names], ['aaa', 'ccc'])
        self.assertEqual([alias.name for alias in t.body[1].names], ['bbb'])
        self.assertEqual(t.body[1].names[0].asname, 'b')

    def test_split_imports_multiple(self):
        src = 'import aaa, bbb, ccc\n'
        t = ast.parse(src)
        import_node = t.body[0]
        alias_bbb = import_node.names[1]
        alias_ccc = import_node.names[2]
        sc = scope.analyze(t)
        import_utils.split_import(sc, import_node, alias_bbb)
        import_utils.split_import(sc, import_node, alias_ccc)
        self.assertEqual(3, len(t.body))
        self.assertEqual([alias.name for alias in t.body[0].names], ['aaa'])
        self.assertEqual([alias.name for alias in t.body[1].names], ['ccc'])
        self.assertEqual([alias.name for alias in t.body[2].names], ['bbb'])

    def test_split_nested_imports(self):
        test_cases = ('def foo():\n  {import_stmt}\n', 'class Foo(object):\n  {import_stmt}\n', 'if foo:\n  {import_stmt}\nelse:\n  pass\n', 'if foo:\n  pass\nelse:\n  {import_stmt}\n', 'if foo:\n  pass\nelif bar:\n  {import_stmt}\n', 'try:\n  {import_stmt}\nexcept:\n  pass\n', 'try:\n  pass\nexcept:\n  {import_stmt}\n', 'try:\n  pass\nfinally:\n  {import_stmt}\n', 'for i in foo:\n  {import_stmt}\n', 'for i in foo:\n  pass\nelse:\n  {import_stmt}\n', 'while foo:\n  {import_stmt}\n')
        for template in test_cases:
            try:
                src = template.format(import_stmt='import aaa, bbb, ccc')
                t = ast.parse(src)
                sc = scope.analyze(t)
                import_node = ast_utils.find_nodes_by_type(t, ast.Import)[0]
                import_utils.split_import(sc, import_node, import_node.names[1])
                split_import_nodes = ast_utils.find_nodes_by_type(t, ast.Import)
                self.assertEqual(1, len(t.body))
                self.assertEqual(2, len(split_import_nodes))
                self.assertEqual([alias.name for alias in split_import_nodes[0].names], ['aaa', 'ccc'])
                self.assertEqual([alias.name for alias in split_import_nodes[1].names], ['bbb'])
            except:
                self.fail('Failed while executing case:\n%s\nCaused by:\n%s' % (src, traceback.format_exc()))