from __future__ import absolute_import
import sys
import os.path
from textwrap import dedent
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from Cython.Compiler.ParseTreeTransforms import NormalizeTree, InterpretCompilerDirectives
from Cython.Compiler import Main, Symtab, Visitor, Options
from Cython.TestUtils import TransformTest
def _test_typing_function_char_loop(self):
    code = "        def func(x):\n            l = []\n            for c in x:\n                l.append(c)\n            return l\n\n        print(func('abcdefg'))\n        "
    types = self._test(code)
    self.assertIn(('func', (1, 0)), types)
    variables = types.pop(('func', (1, 0)))
    self.assertFalse(types)
    self.assertEqual({'a': set(['int']), 'i': set(['int'])}, variables)