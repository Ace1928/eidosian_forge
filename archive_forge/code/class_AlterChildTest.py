from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pasta
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import test_utils
class AlterChildTest(test_utils.TestCase):

    def testRemoveChildMethod(self):
        src = 'class C():\n  def f(x):\n    return x + 2\n  def g(x):\n    return x + 3\n'
        tree = pasta.parse(src)
        class_node = tree.body[0]
        meth1_node = class_node.body[0]
        ast_utils.remove_child(class_node, meth1_node)
        result = pasta.dump(tree)
        expected = 'class C():\n  def g(x):\n    return x + 3\n'
        self.assertEqual(result, expected)

    def testRemoveAlias(self):
        src = 'from a import b, c'
        tree = pasta.parse(src)
        import_node = tree.body[0]
        alias1 = import_node.names[0]
        ast_utils.remove_child(import_node, alias1)
        self.assertEqual(pasta.dump(tree), 'from a import c')

    def testRemoveFromBlock(self):
        src = 'if a:\n  print("foo!")\n  x = 1\n'
        tree = pasta.parse(src)
        if_block = tree.body[0]
        print_stmt = if_block.body[0]
        ast_utils.remove_child(if_block, print_stmt)
        expected = 'if a:\n  x = 1\n'
        self.assertEqual(pasta.dump(tree), expected)

    def testReplaceChildInBody(self):
        src = 'def foo():\n  a = 0\n  a += 1 # replace this\n  return a\n'
        replace_with = pasta.parse('foo(a + 1)  # trailing comment\n').body[0]
        expected = 'def foo():\n  a = 0\n  foo(a + 1) # replace this\n  return a\n'
        t = pasta.parse(src)
        parent = t.body[0]
        node_to_replace = parent.body[1]
        ast_utils.replace_child(parent, node_to_replace, replace_with)
        self.assertEqual(expected, pasta.dump(t))

    def testReplaceChildInvalid(self):
        src = 'def foo():\n  return 1\nx = 1\n'
        replace_with = pasta.parse('bar()').body[0]
        t = pasta.parse(src)
        parent = t.body[0]
        node_to_replace = t.body[1]
        with self.assertRaises(errors.InvalidAstError):
            ast_utils.replace_child(parent, node_to_replace, replace_with)