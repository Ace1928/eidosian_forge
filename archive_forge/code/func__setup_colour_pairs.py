from __future__ import annotations
import curses
import sys
import typing
from contextlib import suppress
from urwid import util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, AttrSpec, BaseScreen, RealTerminal
def _setup_colour_pairs(self) -> None:
    """
        Initialize all 63 color pairs based on the term:
        bg * 8 + 7 - fg
        So to get a color, we just need to use that term and get the right color
        pair number.
        """
    if not self.has_color:
        return
    if IS_WINDOWS:
        self.has_default_colors = False
    for fg in range(8):
        for bg in range(8):
            if fg == curses.COLOR_WHITE and bg == curses.COLOR_BLACK:
                continue
            curses.init_pair(bg * 8 + 7 - fg, COLOR_CORRECTION.get(fg, fg), COLOR_CORRECTION.get(bg, bg))