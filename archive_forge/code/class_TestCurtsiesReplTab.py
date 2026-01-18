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
class TestCurtsiesReplTab(TestCase):

    def setUp(self):
        self.repl = create_repl()
        self.repl.matches_iter = MagicIterMock()

        def add_matches(*args, **kwargs):
            self.repl.matches_iter.matches = ['aaa', 'aab', 'aac']
        self.repl.complete = mock.Mock(side_effect=add_matches, return_value=True)

    def test_tab_with_no_matches_triggers_completion(self):
        self.repl._current_line = ' asdf'
        self.repl._cursor_offset = 5
        self.repl.matches_iter.matches = []
        self.repl.matches_iter.is_cseq.return_value = False
        self.repl.matches_iter.cur_line.return_value = (None, None)
        self.repl.on_tab()
        self.repl.complete.assert_called_once_with(tab=True)

    def test_tab_after_indentation_adds_space(self):
        self.repl._current_line = '    '
        self.repl._cursor_offset = 4
        self.repl.on_tab()
        self.assertEqual(self.repl._current_line, '        ')
        self.assertEqual(self.repl._cursor_offset, 8)

    def test_tab_at_beginning_of_line_adds_space(self):
        self.repl._current_line = ''
        self.repl._cursor_offset = 0
        self.repl.on_tab()
        self.assertEqual(self.repl._current_line, '    ')
        self.assertEqual(self.repl._cursor_offset, 4)

    def test_tab_with_no_matches_selects_first(self):
        self.repl._current_line = ' aa'
        self.repl._cursor_offset = 3
        self.repl.matches_iter.matches = []
        self.repl.matches_iter.is_cseq.return_value = False
        mock_next(self.repl.matches_iter, None)
        self.repl.matches_iter.cur_line.return_value = (None, None)
        self.repl.on_tab()
        self.repl.complete.assert_called_once_with(tab=True)
        self.repl.matches_iter.cur_line.assert_called_once_with()

    def test_tab_with_matches_selects_next_match(self):
        self.repl._current_line = ' aa'
        self.repl._cursor_offset = 3
        self.repl.complete()
        self.repl.matches_iter.is_cseq.return_value = False
        mock_next(self.repl.matches_iter, None)
        self.repl.matches_iter.cur_line.return_value = (None, None)
        self.repl.on_tab()
        self.repl.matches_iter.cur_line.assert_called_once_with()

    def test_tab_completes_common_sequence(self):
        self.repl._current_line = ' a'
        self.repl._cursor_offset = 2
        self.repl.matches_iter.matches = ['aaa', 'aab', 'aac']
        self.repl.matches_iter.is_cseq.return_value = True
        self.repl.matches_iter.substitute_cseq.return_value = (None, None)
        self.repl.on_tab()
        self.repl.matches_iter.substitute_cseq.assert_called_once_with()