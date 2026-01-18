from __future__ import annotations
import atexit
import copy
import errno
import fcntl
import os
import pty
import selectors
import signal
import struct
import sys
import termios
import time
import traceback
import typing
import warnings
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from urwid import event_loop, util
from urwid.canvas import Canvas
from urwid.display import AttrSpec, RealTerminal
from urwid.display.escape import ALT_DEC_SPECIAL_CHARS, DEC_SPECIAL_CHARS
from urwid.widget import Sizing, Widget
from .display.common import _BASIC_COLORS, _color_desc_256, _color_desc_true
@dataclass(eq=True, order=False)
class TermModes:
    display_ctrl: bool = False
    insert: bool = False
    lfnl: bool = False
    keys_decckm: bool = False
    reverse_video: bool = False
    constrain_scrolling: bool = False
    autowrap: bool = True
    visible_cursor: bool = True
    bracketed_paste: bool = False
    main_charset: Literal[1, 2] = CHARSET_DEFAULT

    def reset(self) -> None:
        self.display_ctrl = False
        self.insert = False
        self.lfnl = False
        self.keys_decckm = False
        self.reverse_video = False
        self.constrain_scrolling = False
        self.autowrap = True
        self.visible_cursor = True
        self.main_charset = CHARSET_DEFAULT