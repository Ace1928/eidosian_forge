from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import difflib
import itertools
import os.path
from six import with_metaclass
import sys
import textwrap
import unittest
import pasta
from pasta.base import annotate
from pasta.base import ast_utils
from pasta.base import codegen
from pasta.base import formatting as fmt
from pasta.base import test_utils
class PrefixSuffixTest(test_utils.TestCase):

    def test_block_suffix(self):
        src_tpl = textwrap.dedent('        {open_block}\n          pass #a\n          #b\n            #c\n\n          #d\n        #e\n        a\n        ')
        test_cases = (('body', 'def x():'), ('body', 'class X:'), ('body', 'if x:'), ('orelse', 'if x:\n  y\nelse:'), ('body', 'if x:\n  y\nelif y:'), ('body', 'while x:'), ('orelse', 'while x:\n  y\nelse:'), ('finalbody', 'try:\n  x\nfinally:'), ('body', 'try:\n  x\nexcept:'), ('orelse', 'try:\n  x\nexcept:\n  y\nelse:'), ('body', 'with x:'), ('body', 'with x, y:'), ('body', 'with x:\n with y:'), ('body', 'for x in y:'))

        def is_node_for_suffix(node, children_attr):
            val = getattr(node, children_attr, None)
            return isinstance(val, list) and type(val[0]) == ast.Pass
        for children_attr, open_block in test_cases:
            src = src_tpl.format(open_block=open_block)
            t = pasta.parse(src)
            node_finder = ast_utils.FindNodeVisitor(lambda node: is_node_for_suffix(node, children_attr))
            node_finder.visit(t)
            node = node_finder.results[0]
            expected = '  #b\n    #c\n\n  #d\n'
            actual = str(fmt.get(node, 'block_suffix_%s' % children_attr))
            self.assertMultiLineEqual(expected, actual, 'Incorrect suffix for code:\n%s\nNode: %s (line %d)\nDiff:\n%s' % (src, node, node.lineno, '\n'.join(_get_diff(actual, expected))))
            self.assertMultiLineEqual(src, pasta.dump(t))

    def test_module_suffix(self):
        src = 'foo\n#bar\n\n#baz\n'
        t = pasta.parse(src)
        self.assertEqual(src[src.index('#bar'):], fmt.get(t, 'suffix'))

    def test_no_block_suffix_for_single_line_statement(self):
        src = 'if x:  return y\n  #a\n#b\n'
        t = pasta.parse(src)
        self.assertIsNone(fmt.get(t.body[0], 'block_suffix_body'))

    def test_expression_prefix_suffix(self):
        src = 'a\n\nfoo\n\n\nb\n'
        t = pasta.parse(src)
        self.assertEqual('\n', fmt.get(t.body[1], 'prefix'))
        self.assertEqual('\n', fmt.get(t.body[1], 'suffix'))

    def test_statement_prefix_suffix(self):
        src = 'a\n\ndef foo():\n  return bar\n\n\nb\n'
        t = pasta.parse(src)
        self.assertEqual('\n', fmt.get(t.body[1], 'prefix'))
        self.assertEqual('', fmt.get(t.body[1], 'suffix'))