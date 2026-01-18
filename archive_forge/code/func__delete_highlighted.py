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
def _delete_highlighted(self) -> bool:
    """
        Delete all highlighted text and update cursor position, if any
        text is highlighted.
        """
    if not self.highlight:
        return False
    start, stop = self.highlight
    btext, etext = (self.edit_text[:start], self.edit_text[stop:])
    self.set_edit_text(btext + etext)
    self.edit_pos = start
    self.highlight = None
    return True