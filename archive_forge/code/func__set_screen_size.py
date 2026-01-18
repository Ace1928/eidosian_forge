from __future__ import annotations
import dataclasses
import glob
import html
import os
import pathlib
import random
import selectors
import signal
import socket
import string
import sys
import tempfile
import typing
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width, move_next_char
from urwid.util import StoppingContext, get_encoding
from .common import BaseScreen
def _set_screen_size(self, cols: int, rows: int) -> None:
    """Set the screen size (within max size)."""
    cols = min(cols, MAX_COLS)
    rows = min(rows, MAX_ROWS)
    self.screen_size = (cols, rows)