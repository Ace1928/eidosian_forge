import code
import os
import sys
import tempfile
import io
from typing import cast
import unittest
from contextlib import contextmanager
from functools import partial
from unittest import mock
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython.curtsiesfrontend import interpreter
from bpython.curtsiesfrontend import events as bpythonevents
from bpython.curtsiesfrontend.repl import LineType
from bpython import autocomplete
from bpython import config
from bpython import args
from bpython.test import (
from curtsies import events
from curtsies.window import CursorAwareWindow
from importlib import invalidate_caches
class TestCurtsiesReevaluateWithImport(TestCase):

    def setUp(self):
        self.repl = create_repl()
        self.open = partial(io.open, mode='wt', encoding='utf-8')
        self.dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        self.sys_path = sys.path
        sys.path = self.sys_path[:]
        invalidate_caches()

    def tearDown(self):
        sys.dont_write_bytecode = self.dont_write_bytecode
        sys.path = self.sys_path

    def push(self, line):
        self.repl._current_line = line
        self.repl.on_enter()

    def head(self, path):
        self.push('import sys')
        self.push('sys.path.append("%s")' % path)

    @staticmethod
    @contextmanager
    def tempfile():
        with tempfile.NamedTemporaryFile(suffix='.py') as temp:
            path, name = os.path.split(temp.name)
            yield (temp.name, path, name.replace('.py', ''))

    def test_module_content_changed(self):
        with self.tempfile() as (fullpath, path, modname):
            print(modname)
            with self.open(fullpath) as f:
                f.write('a = 0\n')
            self.head(path)
            self.push('import %s' % modname)
            self.push('a = %s.a' % modname)
            self.assertIn('a', self.repl.interp.locals)
            self.assertEqual(self.repl.interp.locals['a'], 0)
            with self.open(fullpath) as f:
                f.write('a = 1\n')
            self.repl.clear_modules_and_reevaluate()
            self.assertIn('a', self.repl.interp.locals)
            self.assertEqual(self.repl.interp.locals['a'], 1)

    def test_import_module_with_rewind(self):
        with self.tempfile() as (fullpath, path, modname):
            print(modname)
            with self.open(fullpath) as f:
                f.write('a = 0\n')
            self.head(path)
            self.push('import %s' % modname)
            self.assertIn(modname, self.repl.interp.locals)
            self.repl.undo()
            self.assertNotIn(modname, self.repl.interp.locals)
            self.repl.clear_modules_and_reevaluate()
            self.assertNotIn(modname, self.repl.interp.locals)
            self.push('import %s' % modname)
            self.push('a = %s.a' % modname)
            self.assertIn('a', self.repl.interp.locals)
            self.assertEqual(self.repl.interp.locals['a'], 0)
            with self.open(fullpath) as f:
                f.write('a = 1\n')
            self.repl.clear_modules_and_reevaluate()
            self.assertIn('a', self.repl.interp.locals)
            self.assertEqual(self.repl.interp.locals['a'], 1)