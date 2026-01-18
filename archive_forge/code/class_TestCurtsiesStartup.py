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
class TestCurtsiesStartup(TestCase):

    def setUp(self):
        self.repl = create_repl()

    def write_startup_file(self, fname, encoding):
        with open(fname, mode='wt', encoding=encoding) as f:
            f.write('# coding: ')
            f.write(encoding)
            f.write('\n')
            f.write('a = "äöü"\n')

    def test_startup_event_utf8(self):
        with tempfile.NamedTemporaryFile() as temp:
            self.write_startup_file(temp.name, 'utf-8')
            with mock.patch.dict('os.environ', {'PYTHONSTARTUP': temp.name}):
                self.repl.process_event(bpythonevents.RunStartupFileEvent())
        self.assertIn('a', self.repl.interp.locals)

    def test_startup_event_latin1(self):
        with tempfile.NamedTemporaryFile() as temp:
            self.write_startup_file(temp.name, 'latin-1')
            with mock.patch.dict('os.environ', {'PYTHONSTARTUP': temp.name}):
                self.repl.process_event(bpythonevents.RunStartupFileEvent())
        self.assertIn('a', self.repl.interp.locals)