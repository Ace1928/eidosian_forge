import curses
import errno
import functools
import math
import os
import platform
import re
import struct
import sys
import time
from typing import (
from ._typing_compat import Literal
import unicodedata
from dataclasses import dataclass
from pygments import format
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from pygments.token import Token, _TokenType
from .formatter import BPythonFormatter
from .config import getpreferredencoding, Config
from .keys import cli_key_dispatch as key_dispatch
from . import translations
from .translations import _
from . import repl, inspection
from . import args as bpargs
from .pager import page
from .args import parse as argsparse
class Statusbar:
    """This class provides the status bar at the bottom of the screen.
    It has message() and prompt() methods for user interactivity, as
    well as settext() and clear() methods for changing its appearance.

    The check() method needs to be called repeatedly if the statusbar is
    going to be aware of when it should update its display after a message()
    has been called (it'll display for a couple of seconds and then disappear).

    It should be called as:
        foo = Statusbar(stdscr, scr, 'Initial text to display')
    or, for a blank statusbar:
        foo = Statusbar(stdscr, scr)

    It can also receive the argument 'c' which will be an integer referring
    to a curses colour pair, e.g.:
        foo = Statusbar(stdscr, 'Hello', c=4)

    stdscr should be a curses window object in which to put the status bar.
    pwin should be the parent window. To be honest, this is only really here
    so the cursor can be returned to the window properly.

    """

    def __init__(self, scr: '_CursesWindow', pwin: '_CursesWindow', background: int, config: Config, s: Optional[str]=None, c: Optional[int]=None):
        """Initialise the statusbar and display the initial text (if any)"""
        self.size()
        self.win: '_CursesWindow' = newwin(background, self.h, self.w, self.y, self.x)
        self.config = config
        self.s = s or ''
        self._s = self.s
        self.c = c
        self.timer = 0
        self.pwin = pwin
        if s:
            self.settext(s, c)

    def size(self) -> None:
        """Set instance attributes for x and y top left corner coordinates
        and width and height for the window."""
        h, w = gethw()
        self.y = h - 1
        self.w = w
        self.h = 1
        self.x = 0

    def resize(self, refresh: bool=True) -> None:
        """This method exists simply to keep it straight forward when
        initialising a window and resizing it."""
        self.size()
        self.win.mvwin(self.y, self.x)
        self.win.resize(self.h, self.w)
        if refresh:
            self.refresh()

    def refresh(self) -> None:
        """This is here to make sure the status bar text is redraw properly
        after a resize."""
        self.settext(self._s)

    def check(self) -> None:
        """This is the method that should be called every half second or so
        to see if the status bar needs updating."""
        if not self.timer:
            return
        if time.time() < self.timer:
            return
        self.settext(self._s)

    def message(self, s: str, n: float=3.0) -> None:
        """Display a message for a short n seconds on the statusbar and return
        it to its original state."""
        self.timer = int(time.time() + n)
        self.settext(s)

    def prompt(self, s: str='') -> str:
        """Prompt the user for some input (with the optional prompt 's') and
        return the input text, then restore the statusbar to its original
        value."""
        self.settext(s or '? ', p=True)
        iy, ix = self.win.getyx()

        def bs(s: str) -> str:
            y, x = self.win.getyx()
            if x == ix:
                return s
            s = s[:-1]
            self.win.delch(y, x - 1)
            self.win.move(y, x - 1)
            return s
        o = ''
        while True:
            c = self.win.getch()
            if c == 127:
                o = bs(o)
            elif c == 10:
                break
            elif c == 27:
                curses.flushinp()
                raise ValueError
            elif 0 < c < 127:
                d = chr(c)
                self.win.addstr(d, get_colpair(self.config, 'prompt'))
                o += d
        self.settext(self._s)
        return o

    def settext(self, s: str, c: Optional[int]=None, p: bool=False) -> None:
        """Set the text on the status bar to a new permanent value; this is the
        value that will be set after a prompt or message. c is the optional
        curses colour pair to use (if not specified the last specified colour
        pair will be used).  p is True if the cursor is expected to stay in the
        status window (e.g. when prompting)."""
        self.win.erase()
        if len(s) >= self.w:
            s = s[:self.w - 1]
        self.s = s
        if c:
            self.c = c
        if s:
            if self.c:
                self.win.addstr(s, self.c)
            else:
                self.win.addstr(s)
        if not p:
            self.win.noutrefresh()
            self.pwin.refresh()
        else:
            self.win.refresh()

    def clear(self) -> None:
        """Clear the status bar."""
        self.win.clear()