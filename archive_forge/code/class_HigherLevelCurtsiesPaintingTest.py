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
class HigherLevelCurtsiesPaintingTest(CurtsiesPaintingTest):

    def refresh(self):
        self.refresh_requests.append(RefreshRequestEvent())

    def send_refreshes(self):
        while self.refresh_requests:
            self.repl.process_event(self.refresh_requests.pop())
            _, _ = self.repl.paint()

    def enter(self, line=None):
        """Enter a line of text, avoiding autocompletion windows

        autocomplete could still happen if the entered line has
        autocompletion that would happen then, but intermediate
        stages won't happen"""
        if line is not None:
            self.repl._set_cursor_offset(len(line), update_completion=False)
            self.repl.current_line = line
        with output_to_repl(self.repl):
            self.repl.on_enter(new_code=False)
            self.assertEqual(self.repl.rl_history.entries, [''])
            self.send_refreshes()

    def undo(self):
        with output_to_repl(self.repl):
            self.repl.undo()
            self.send_refreshes()

    def setUp(self):
        self.refresh_requests = []

        class TestRepl(BaseRepl):

            def _request_refresh(inner_self):
                self.refresh()
        self.repl = TestRepl(setup_config(), cast(CursorAwareWindow, None), banner='')
        self.repl.height, self.repl.width = (5, 32)

    def send_key(self, key):
        self.repl.process_event('<SPACE>' if key == ' ' else key)
        self.repl.paint()