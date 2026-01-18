from __future__ import absolute_import
import sys
import os.path
from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor, Options
from Cython.TestUtils import TransformTest
class TestJediTyper(TransformTest):

    def _test(self, code):
        return _test_typing(code)[0]

    def test_typing_global_int_loop(self):
        code = '        for i in range(10):\n            a = i + 1\n        '
        types = self._test(code)
        self.assertIn((None, (1, 0)), types)
        variables = types.pop((None, (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['int']), 'i': set(['int'])}, variables)

    def test_typing_function_int_loop(self):
        code = '        def func(x):\n            for i in range(x):\n                a = i + 1\n            return a\n        '
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['int']), 'i': set(['int'])}, variables)

    def test_conflicting_types_in_function(self):
        code = "        def func(a, b):\n            print(a)\n            a = 1\n            b += a\n            a = 'abc'\n            return a, str(b)\n\n        print(func(1.5, 2))\n        "
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['float', 'int', 'str']), 'b': set(['int'])}, variables)

    def _test_typing_function_char_loop(self):
        code = "        def func(x):\n            l = []\n            for c in x:\n                l.append(c)\n            return l\n\n        print(func('abcdefg'))\n        "
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['int']), 'i': set(['int'])}, variables)

    def test_typing_global_list(self):
        code = '        a = [x for x in range(10)]\n        b = list(range(10))\n        c = a + b\n        d = [0]*10\n        '
        types = self._test(code)
        self.assertIn((None, (1, 0)), types)
        variables = types.pop((None, (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['list']), 'b': set(['list']), 'c': set(['list']), 'd': set(['list'])}, variables)

    def test_typing_function_list(self):
        code = '        def func(x):\n            a = [[], []]\n            b = [0]* 10 + a\n            c = a[0]\n\n        print(func([0]*100))\n        '
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['list']), 'b': set(['list']), 'c': set(['list']), 'x': set(['list'])}, variables)

    def test_typing_global_dict(self):
        code = '        a = dict()\n        b = {i: i**2 for i in range(10)}\n        c = a\n        '
        types = self._test(code)
        self.assertIn((None, (1, 0)), types)
        variables = types.pop((None, (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['dict']), 'b': set(['dict']), 'c': set(['dict'])}, variables)

    def test_typing_function_dict(self):
        code = "        def func(x):\n            a = dict()\n            b = {i: i**2 for i in range(10)}\n            c = x\n\n        print(func({1:2, 'x':7}))\n        "
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['dict']), 'b': set(['dict']), 'c': set(['dict']), 'x': set(['dict'])}, variables)

    def test_typing_global_set(self):
        code = '        a = set()\n        # b = {i for i in range(10)} # jedi does not support set comprehension yet\n        c = a\n        d = {1,2,3}\n        e = a | b\n        '
        types = self._test(code)
        self.assertIn((None, (1, 0)), types)
        variables = types.pop((None, (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['set']), 'c': set(['set']), 'd': set(['set']), 'e': set(['set'])}, variables)

    def test_typing_function_set(self):
        code = '        def func(x):\n            a = set()\n            # b = {i for i in range(10)} # jedi does not support set comprehension yet\n            c = a\n            d = a | b\n\n        print(func({1,2,3}))\n        '
        types = self._test(code)
        self.assertIn(('func', (1, 0)), types)
        variables = types.pop(('func', (1, 0)))
        self.assertFalse(types)
        self.assertEqual({'a': set(['set']), 'c': set(['set']), 'd': set(['set']), 'x': set(['set'])}, variables)