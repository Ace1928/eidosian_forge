from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import textwrap
import unittest
from pasta.augment import inline
from pasta.base import test_utils
class InlineTest(test_utils.TestCase):

    def test_inline_simple(self):
        src = 'x = 1\na = x\n'
        t = ast.parse(src)
        inline.inline_name(t, 'x')
        self.checkAstsEqual(t, ast.parse('a = 1\n'))

    def test_inline_multiple_targets(self):
        src = 'x = y = z = 1\na = x + y\n'
        t = ast.parse(src)
        inline.inline_name(t, 'y')
        self.checkAstsEqual(t, ast.parse('x = z = 1\na = x + 1\n'))

    def test_inline_multiple_reads(self):
        src = textwrap.dedent('        CONSTANT = "foo"\n        def a(b=CONSTANT):\n          return b == CONSTANT\n        ')
        expected = textwrap.dedent('        def a(b="foo"):\n          return b == "foo"\n        ')
        t = ast.parse(src)
        inline.inline_name(t, 'CONSTANT')
        self.checkAstsEqual(t, ast.parse(expected))

    def test_inline_non_constant_fails(self):
        src = textwrap.dedent('        NOT_A_CONSTANT = "foo"\n        NOT_A_CONSTANT += "bar"\n        ')
        t = ast.parse(src)
        with self.assertRaisesRegexp(inline.InlineError, "'NOT_A_CONSTANT' is not a constant"):
            inline.inline_name(t, 'NOT_A_CONSTANT')

    def test_inline_function_fails(self):
        src = 'def func(): pass\nfunc()\n'
        t = ast.parse(src)
        with self.assertRaisesRegexp(inline.InlineError, "'func' is not a constant; it has type %r" % ast.FunctionDef):
            inline.inline_name(t, 'func')

    def test_inline_conditional_fails(self):
        src = 'if define:\n  x = 1\na = x\n'
        t = ast.parse(src)
        with self.assertRaisesRegexp(inline.InlineError, "'x' is not a top-level name"):
            inline.inline_name(t, 'x')

    def test_inline_non_assign_fails(self):
        src = 'CONSTANT1, CONSTANT2 = values'
        t = ast.parse(src)
        with self.assertRaisesRegexp(inline.InlineError, "'CONSTANT1' is not declared in an assignment"):
            inline.inline_name(t, 'CONSTANT1')