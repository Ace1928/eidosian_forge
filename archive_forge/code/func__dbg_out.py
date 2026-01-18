from __future__ import annotations
import curses
import sys
import typing
from contextlib import suppress
from urwid import util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, AttrSpec, BaseScreen, RealTerminal
def _dbg_out(self, string) -> None:
    self.s.clrtoeol()
    self.s.addstr(string)
    self.s.refresh()
    self._curs_set(1)