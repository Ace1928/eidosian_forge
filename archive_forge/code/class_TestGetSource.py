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
class TestGetSource(unittest.TestCase):

    def setUp(self):
        self.repl = FakeRepl()

    def set_input_line(self, line):
        """Set current input line of the test REPL."""
        self.repl.current_line = line
        self.repl.cursor_offset = len(line)

    def assert_get_source_error_for_current_function(self, func, msg):
        self.repl.current_func = func
        with self.assertRaises(repl.SourceNotFound):
            self.repl.get_source_of_current_name()
        try:
            self.repl.get_source_of_current_name()
        except repl.SourceNotFound as e:
            self.assertEqual(e.args[0], msg)
        else:
            self.fail('Should have raised SourceNotFound')

    def test_current_function(self):
        self.set_input_line('INPUTLINE')
        self.repl.current_func = inspect.getsource
        self.assertIn('text of the source code', self.repl.get_source_of_current_name())
        self.assert_get_source_error_for_current_function([], 'No source code found for INPUTLINE')
        self.assert_get_source_error_for_current_function(list.pop, 'No source code found for INPUTLINE')

    @unittest.skipIf(pypy, 'different errors for PyPy')
    def test_current_function_cpython(self):
        self.set_input_line('INPUTLINE')
        self.assert_get_source_error_for_current_function(collections.defaultdict.copy, 'No source code found for INPUTLINE')
        self.assert_get_source_error_for_current_function(collections.defaultdict, 'could not find class definition')

    def test_current_line(self):
        self.repl.interp.locals['a'] = socket.socket
        self.set_input_line('a')
        self.assertIn('dup(self)', self.repl.get_source_of_current_name())