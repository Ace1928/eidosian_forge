from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
class CanvasError(Exception):
    pass