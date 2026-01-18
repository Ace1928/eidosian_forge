import itertools
import os
import pydoc
import string
import sys
from contextlib import contextmanager
from typing import cast
from curtsies.formatstringarray import (
from curtsies.fmtfuncs import cyan, bold, green, yellow, on_magenta, red
from curtsies.window import CursorAwareWindow
from unittest import mock, skipIf
from bpython.curtsiesfrontend.events import RefreshRequestEvent
from bpython import config, inspection
from bpython.curtsiesfrontend.repl import BaseRepl
from bpython.curtsiesfrontend import replpainter
from bpython.curtsiesfrontend.repl import (
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
class CurtsiesPaintingTest(ClearEnviron):

    def setUp(self):

        class TestRepl(BaseRepl):

            def _request_refresh(inner_self):
                pass
        self.repl = TestRepl(setup_config(), cast(CursorAwareWindow, None))
        self.repl.height, self.repl.width = (5, 10)

    @property
    def locals(self):
        return self.repl.coderunner.interp.locals

    def assert_paint(self, screen, cursor_row_col):
        array, cursor_pos = self.repl.paint()
        assertFSArraysEqual(array, screen)
        self.assertEqual(cursor_pos, cursor_row_col)

    def assert_paint_ignoring_formatting(self, screen, cursor_row_col=None, **paint_kwargs):
        array, cursor_pos = self.repl.paint(**paint_kwargs)
        assertFSArraysEqualIgnoringFormatting(array, screen)
        if cursor_row_col is not None:
            self.assertEqual(cursor_pos, cursor_row_col)

    def process_box_characters(self, screen):
        if not self.repl.config.unicode_box or not config.supports_box_chars():
            return [line.replace('┌', '+').replace('└', '+').replace('┘', '+').replace('┐', '+').replace('─', '-') for line in screen]
        return screen