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
class TestCurtsiesPaintingSimple(CurtsiesPaintingTest):

    def test_startup(self):
        screen = fsarray([cyan('>>> '), cyan('Welcome to')])
        self.assert_paint(screen, (0, 4))

    def test_enter_text(self):
        [self.repl.add_normal_character(c) for c in '1 + 1']
        screen = fsarray([cyan('>>> ') + bold(green('1') + cyan(' ') + yellow('+') + cyan(' ') + green('1')), cyan('Welcome to')])
        self.assert_paint(screen, (0, 9))

    def test_run_line(self):
        try:
            orig_stdout = sys.stdout
            sys.stdout = self.repl.stdout
            [self.repl.add_normal_character(c) for c in '1 + 1']
            self.repl.on_enter(new_code=False)
            screen = fsarray(['>>> 1 + 1', '2', 'Welcome to'])
            self.assert_paint_ignoring_formatting(screen, (1, 1))
        finally:
            sys.stdout = orig_stdout

    def test_completion(self):
        self.repl.height, self.repl.width = (5, 32)
        self.repl.current_line = 'an'
        self.cursor_offset = 2
        screen = self.process_box_characters(['>>> an', '┌──────────────────────────────┐', '│ and  any(                    │', '└──────────────────────────────┘', 'Welcome to bpython! Press <F1> f'] if sys.version_info[:2] < (3, 10) else ['>>> an', '┌──────────────────────────────┐', '│ and    anext( any(           │', '└──────────────────────────────┘', 'Welcome to bpython! Press <F1> f'])
        self.assert_paint_ignoring_formatting(screen, (0, 4))

    def test_argspec(self):

        def foo(x, y, z=10):
            """docstring!"""
            pass
        argspec = inspection.getfuncprops('foo', foo)
        array = replpainter.formatted_argspec(argspec, 1, 30, setup_config())
        screen = [bold(cyan('foo')) + cyan(':') + cyan(' ') + cyan('(') + cyan('x') + yellow(',') + yellow(' ') + bold(cyan('y')) + yellow(',') + yellow(' ') + cyan('z') + yellow('=') + bold(cyan('10')) + yellow(')')]
        assertFSArraysEqual(fsarray(array), fsarray(screen))

    def test_formatted_docstring(self):
        actual = replpainter.formatted_docstring('Returns the results\n\nAlso has side effects', 40, config=setup_config())
        expected = fsarray(['Returns the results', '', 'Also has side effects'])
        assertFSArraysEqualIgnoringFormatting(actual, expected)

    def test_unicode_docstrings(self):
        """A bit of a special case in Python 2"""

        def foo():
            """åß∂ƒ"""
        actual = replpainter.formatted_docstring(foo.__doc__, 40, config=setup_config())
        expected = fsarray(['åß∂ƒ'])
        assertFSArraysEqualIgnoringFormatting(actual, expected)

    def test_nonsense_docstrings(self):
        for docstring in [123, {}, []]:
            try:
                replpainter.formatted_docstring(docstring, 40, config=setup_config())
            except Exception:
                self.fail(f'bad docstring caused crash: {docstring!r}')

    def test_weird_boto_docstrings(self):

        class WeirdDocstring(str):

            def expandtabs(self, tabsize=8):
                return 'asdfåß∂ƒ'.expandtabs(tabsize)

        def foo():
            pass
        foo.__doc__ = WeirdDocstring()
        wd = pydoc.getdoc(foo)
        actual = replpainter.formatted_docstring(wd, 40, config=setup_config())
        expected = fsarray(['asdfåß∂ƒ'])
        assertFSArraysEqualIgnoringFormatting(actual, expected)

    def test_paint_lasts_events(self):
        actual = replpainter.paint_last_events(4, 100, ['a', 'b', 'c'], config=setup_config())
        if config.supports_box_chars():
            expected = fsarray(['┌─┐', '│c│', '│b│', '└─┘'])
        else:
            expected = fsarray(['+-+', '|c|', '|b|', '+-+'])
        assertFSArraysEqualIgnoringFormatting(actual, expected)