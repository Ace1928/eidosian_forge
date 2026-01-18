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
def _defaulter(color: int | None, colors: int) -> str:
    if color is None:
        return 'default'
    if color > 255 or colors == 2 ** 24:
        return _color_desc_true(color)
    if color > 15 or colors == 256:
        return _color_desc_256(color)
    return _BASIC_COLORS[color]