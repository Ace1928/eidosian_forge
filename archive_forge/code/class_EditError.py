from __future__ import annotations
import string
import typing
from urwid import text_layout
from urwid.canvas import CompositeCanvas
from urwid.command_map import Command
from urwid.split_repr import remove_defaults
from urwid.str_util import is_wide_char, move_next_char, move_prev_char
from urwid.util import decompose_tagmarkup
from .constants import Align, Sizing, WrapMode
from .text import Text, TextError
class EditError(TextError):
    pass