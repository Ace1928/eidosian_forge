import os
from typing import cast
from bpython.test import FixLanguageTestCase as TestCase, TEST_CONFIG
from bpython.curtsiesfrontend import repl as curtsiesrepl
from bpython import config
from curtsies.window import CursorAwareWindow
class TestBracketCompletionDisabled(TestCase):

    def setUp(self):
        self.repl = create_repl(brackets_enabled=False)

    def process_multiple_events(self, event_list):
        for event in event_list:
            self.repl.process_event(event)

    def test_start_line(self):
        self.repl.process_event('(')
        self.assertEqual(self.repl._current_line, '(')
        self.assertEqual(self.repl._cursor_offset, 1)

    def test_nested_brackets(self):
        self.process_multiple_events(['(', '[', '{'])
        self.assertEqual(self.repl._current_line, '([{')
        self.assertEqual(self.repl._cursor_offset, 3)

    def test_bracket_overwrite_closing_char(self):
        self.process_multiple_events(['(', '[', '{'])
        self.assertEqual(self.repl._current_line, '([{')
        self.assertEqual(self.repl._cursor_offset, 3)
        self.process_multiple_events(['}', ']', ')'])
        self.assertEqual(self.repl._current_line, '([{}])')
        self.assertEqual(self.repl._cursor_offset, 6)

    def test_brackets_move_cursor_on_tab(self):
        self.process_multiple_events(['(', '[', '{'])
        self.assertEqual(self.repl._current_line, '([{')
        self.assertEqual(self.repl._cursor_offset, 3)
        self.repl.process_event('<TAB>')
        self.assertEqual(self.repl._current_line, '([{')
        self.assertEqual(self.repl._cursor_offset, 3)

    def test_brackets_deletion_on_backspace(self):
        self.repl.current_line = 'def foo()'
        self.repl.cursor_offset = 8
        self.repl.process_event('<BACKSPACE>')
        self.assertEqual(self.repl._current_line, 'def foo')
        self.assertEqual(self.repl.cursor_offset, 7)

    def test_brackets_deletion_on_backspace_nested(self):
        self.repl.current_line = '([{""}])'
        self.repl.cursor_offset = 4
        self.process_multiple_events(['<BACKSPACE>', '<BACKSPACE>', '<BACKSPACE>'])
        self.assertEqual(self.repl._current_line, '()')
        self.assertEqual(self.repl.cursor_offset, 1)