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
class RemoveImportTest(test_utils.TestCase):

    def test_remove_just_alias(self):
        src = 'import a, b'
        tree = ast.parse(src)
        sc = scope.analyze(tree)
        unused_b_node = tree.body[0].names[1]
        import_utils.remove_import_alias_node(sc, unused_b_node)
        self.assertEqual(len(tree.body), 1)
        self.assertEqual(type(tree.body[0]), ast.Import)
        self.assertEqual(len(tree.body[0].names), 1)
        self.assertEqual(tree.body[0].names[0].name, 'a')

    def test_remove_just_alias_import_from(self):
        src = 'from m import a, b'
        tree = ast.parse(src)
        sc = scope.analyze(tree)
        unused_b_node = tree.body[0].names[1]
        import_utils.remove_import_alias_node(sc, unused_b_node)
        self.assertEqual(len(tree.body), 1)
        self.assertEqual(type(tree.body[0]), ast.ImportFrom)
        self.assertEqual(len(tree.body[0].names), 1)
        self.assertEqual(tree.body[0].names[0].name, 'a')

    def test_remove_full_import(self):
        src = 'import a'
        tree = ast.parse(src)
        sc = scope.analyze(tree)
        a_node = tree.body[0].names[0]
        import_utils.remove_import_alias_node(sc, a_node)
        self.assertEqual(len(tree.body), 0)

    def test_remove_full_importfrom(self):
        src = 'from m import a'
        tree = ast.parse(src)
        sc = scope.analyze(tree)
        a_node = tree.body[0].names[0]
        import_utils.remove_import_alias_node(sc, a_node)
        self.assertEqual(len(tree.body), 0)