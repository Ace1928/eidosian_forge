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
class IndentationTest(test_utils.TestCase):

    def test_indent_levels(self):
        src = textwrap.dedent("        foo('begin')\n        if a:\n          foo('a1')\n          if b:\n            foo('b1')\n            if c:\n              foo('c1')\n            foo('b2')\n          foo('a2')\n        foo('end')\n        ")
        t = pasta.parse(src)
        call_nodes = ast_utils.find_nodes_by_type(t, (ast.Call,))
        call_nodes.sort(key=lambda node: node.lineno)
        begin, a1, b1, c1, b2, a2, end = call_nodes
        self.assertEqual('', fmt.get(begin, 'indent'))
        self.assertEqual('  ', fmt.get(a1, 'indent'))
        self.assertEqual('    ', fmt.get(b1, 'indent'))
        self.assertEqual('      ', fmt.get(c1, 'indent'))
        self.assertEqual('    ', fmt.get(b2, 'indent'))
        self.assertEqual('  ', fmt.get(a2, 'indent'))
        self.assertEqual('', fmt.get(end, 'indent'))

    def test_indent_levels_same_line(self):
        src = 'if a: b; c\n'
        t = pasta.parse(src)
        if_node = t.body[0]
        b, c = if_node.body
        self.assertIsNone(fmt.get(b, 'indent_diff'))
        self.assertIsNone(fmt.get(c, 'indent_diff'))

    def test_indent_depths(self):
        template = 'if a:\n{first}if b:\n{first}{second}foo()\n'
        indents = (' ', ' ' * 2, ' ' * 4, ' ' * 8, '\t', '\t' * 2)
        for first, second in itertools.product(indents, indents):
            src = template.format(first=first, second=second)
            t = pasta.parse(src)
            outer_if_node = t.body[0]
            inner_if_node = outer_if_node.body[0]
            call_node = inner_if_node.body[0]
            self.assertEqual('', fmt.get(outer_if_node, 'indent'))
            self.assertEqual('', fmt.get(outer_if_node, 'indent_diff'))
            self.assertEqual(first, fmt.get(inner_if_node, 'indent'))
            self.assertEqual(first, fmt.get(inner_if_node, 'indent_diff'))
            self.assertEqual(first + second, fmt.get(call_node, 'indent'))
            self.assertEqual(second, fmt.get(call_node, 'indent_diff'))

    def test_indent_multiline_string(self):
        src = textwrap.dedent('        class A:\n          """Doc\n             string."""\n          pass\n        ')
        t = pasta.parse(src)
        docstring, pass_stmt = t.body[0].body
        self.assertEqual('  ', fmt.get(docstring, 'indent'))
        self.assertEqual('  ', fmt.get(pass_stmt, 'indent'))

    def test_indent_multiline_string_with_newline(self):
        src = textwrap.dedent('        class A:\n          """Doc\n\n             string."""\n          pass\n        ')
        t = pasta.parse(src)
        docstring, pass_stmt = t.body[0].body
        self.assertEqual('  ', fmt.get(docstring, 'indent'))
        self.assertEqual('  ', fmt.get(pass_stmt, 'indent'))

    def test_scope_trailing_comma(self):
        template = 'def foo(a, b{trailing_comma}): pass'
        for trailing_comma in ('', ',', ' , '):
            tree = pasta.parse(template.format(trailing_comma=trailing_comma))
            self.assertEqual(trailing_comma.lstrip(' ') + ')', fmt.get(tree.body[0], 'args_suffix'))
        template = 'class Foo(a, b{trailing_comma}): pass'
        for trailing_comma in ('', ',', ' , '):
            tree = pasta.parse(template.format(trailing_comma=trailing_comma))
            self.assertEqual(trailing_comma.lstrip(' ') + ')', fmt.get(tree.body[0], 'bases_suffix'))
        template = 'from mod import (a, b{trailing_comma})'
        for trailing_comma in ('', ',', ' , '):
            tree = pasta.parse(template.format(trailing_comma=trailing_comma))
            self.assertEqual(trailing_comma + ')', fmt.get(tree.body[0], 'names_suffix'))

    def test_indent_extra_newlines(self):
        src = textwrap.dedent('        if a:\n\n          b\n        ')
        t = pasta.parse(src)
        if_node = t.body[0]
        b = if_node.body[0]
        self.assertEqual('  ', fmt.get(b, 'indent_diff'))

    def test_indent_extra_newlines_with_comment(self):
        src = textwrap.dedent('        if a:\n            #not here\n\n          b\n        ')
        t = pasta.parse(src)
        if_node = t.body[0]
        b = if_node.body[0]
        self.assertEqual('  ', fmt.get(b, 'indent_diff'))

    def test_autoindent(self):
        src = textwrap.dedent('        def a():\n            b\n            c\n        ')
        expected = textwrap.dedent('        def a():\n            b\n            new_node\n        ')
        t = pasta.parse(src)
        t.body[0].body[1] = ast.Expr(ast.Name(id='new_node'))
        self.assertMultiLineEqual(expected, codegen.to_str(t))

    @test_utils.requires_features('mixed_tabs_spaces')
    def test_mixed_tabs_spaces_indentation(self):
        pasta.parse(textwrap.dedent('        if a:\n                b\n        {ONETAB}c\n        ').format(ONETAB='\t'))

    @test_utils.requires_features('mixed_tabs_spaces')
    def test_tab_below_spaces(self):
        for num_spaces in range(1, 8):
            t = pasta.parse(textwrap.dedent('          if a:\n          {WS}if b:\n          {ONETAB}c\n          ').format(ONETAB='\t', WS=' ' * num_spaces))
            node_c = t.body[0].body[0].body[0]
            self.assertEqual(fmt.get(node_c, 'indent_diff'), ' ' * (8 - num_spaces))

    @test_utils.requires_features('mixed_tabs_spaces')
    def test_tabs_below_spaces_and_tab(self):
        for num_spaces in range(1, 8):
            t = pasta.parse(textwrap.dedent('          if a:\n          {WS}{ONETAB}if b:\n          {ONETAB}{ONETAB}c\n          ').format(ONETAB='\t', WS=' ' * num_spaces))
            node_c = t.body[0].body[0].body[0]
            self.assertEqual(fmt.get(node_c, 'indent_diff'), '\t')