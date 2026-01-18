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
class TestPredictedIndent(TestCase):

    def setUp(self):
        self.repl = create_repl()

    def test_simple(self):
        self.assertEqual(self.repl.predicted_indent(''), 0)
        self.assertEqual(self.repl.predicted_indent('class Foo:'), 4)
        self.assertEqual(self.repl.predicted_indent('class Foo: pass'), 0)
        self.assertEqual(self.repl.predicted_indent('def asdf():'), 4)
        self.assertEqual(self.repl.predicted_indent('def asdf(): return 7'), 0)

    @unittest.skip('This would be interesting')
    def test_complex(self):
        self.assertEqual(self.repl.predicted_indent('[a, '), 1)
        self.assertEqual(self.repl.predicted_indent('reduce(asdfasdf, '), 7)