import collections
import inspect
import socket
import sys
import tempfile
import unittest
from typing import List, Tuple
from itertools import islice
from pathlib import Path
from unittest import mock
from bpython import config, repl, cli, autocomplete
from bpython.line import LinePart
from bpython.test import (
class TestArgspec(unittest.TestCase):

    def setUp(self):
        self.repl = FakeRepl()
        self.repl.push('def spam(a, b, c):\n', False)
        self.repl.push('    pass\n', False)
        self.repl.push('\n', False)
        self.repl.push('class Spam(object):\n', False)
        self.repl.push('    def spam(self, a, b, c):\n', False)
        self.repl.push('        pass\n', False)
        self.repl.push('\n', False)
        self.repl.push('class SpammitySpam(object):\n', False)
        self.repl.push('    def __init__(self, a, b, c):\n', False)
        self.repl.push('        pass\n', False)
        self.repl.push('\n', False)
        self.repl.push('class WonderfulSpam(object):\n', False)
        self.repl.push('    def __new__(self, a, b, c):\n', False)
        self.repl.push('        pass\n', False)
        self.repl.push('\n', False)
        self.repl.push('o = Spam()\n', False)
        self.repl.push('\n', False)

    def set_input_line(self, line):
        """Set current input line of the test REPL."""
        self.repl.current_line = line
        self.repl.cursor_offset = len(line)

    def test_func_name(self):
        for line, expected_name in [('spam(', 'spam'), ('spam(any([]', 'any') if pypy else ('spam(map([]', 'map'), ('spam((), ', 'spam')]:
            self.set_input_line(line)
            self.assertTrue(self.repl.get_args())
            self.assertEqual(self.repl.current_func.__name__, expected_name)

    def test_func_name_method_issue_479(self):
        for line, expected_name in [('o.spam(', 'spam'), ('o.spam(any([]', 'any') if pypy else ('o.spam(map([]', 'map'), ('o.spam((), ', 'spam')]:
            self.set_input_line(line)
            self.assertTrue(self.repl.get_args())
            self.assertEqual(self.repl.current_func.__name__, expected_name)

    def test_syntax_error_parens(self):
        for line in ['spam(]', 'spam([)', 'spam())']:
            self.set_input_line(line)
            self.repl.get_args()

    def test_kw_arg_position(self):
        self.set_input_line('spam(a=0')
        self.assertTrue(self.repl.get_args())
        self.assertEqual(self.repl.arg_pos, 'a')
        self.set_input_line('spam(1, b=1')
        self.assertTrue(self.repl.get_args())
        self.assertEqual(self.repl.arg_pos, 'b')
        self.set_input_line('spam(1, c=2')
        self.assertTrue(self.repl.get_args())
        self.assertEqual(self.repl.arg_pos, 'c')

    def test_lambda_position(self):
        self.set_input_line('spam(lambda a, b: 1, ')
        self.assertTrue(self.repl.get_args())
        self.assertTrue(self.repl.funcprops)
        self.assertEqual(self.repl.arg_pos, 1)

    @unittest.skipIf(pypy, 'range pydoc has no signature in pypy')
    def test_issue127(self):
        self.set_input_line('x=range(')
        self.assertTrue(self.repl.get_args())
        self.assertEqual(self.repl.current_func.__name__, 'range')
        self.set_input_line('{x:range(')
        self.assertTrue(self.repl.get_args())
        self.assertEqual(self.repl.current_func.__name__, 'range')
        self.set_input_line('foo(1, 2, x,range(')
        self.assertEqual(self.repl.current_func.__name__, 'range')
        self.set_input_line('(x,range(')
        self.assertEqual(self.repl.current_func.__name__, 'range')

    def test_nonexistent_name(self):
        self.set_input_line('spamspamspam(')
        self.assertFalse(self.repl.get_args())

    def test_issue572(self):
        self.set_input_line('SpammitySpam(')
        self.assertTrue(self.repl.get_args())
        self.set_input_line('WonderfulSpam(')
        self.assertTrue(self.repl.get_args())

    @unittest.skipIf(pypy, "pypy pydoc doesn't have this")
    def test_issue583(self):
        self.repl = FakeRepl()
        self.repl.push('a = 1.2\n', False)
        self.set_input_line('a.is_integer(')
        self.repl.set_docstring()
        self.assertIsNot(self.repl.docstring, None)

    def test_methods_of_expressions(self):
        self.set_input_line("'a'.capitalize(")
        self.assertTrue(self.repl.get_args())
        self.set_input_line('(1 + 1.1).as_integer_ratio(')
        self.assertTrue(self.repl.get_args())