from __future__ import annotations
import typing
import warnings
from pprint import pformat
from urwid.canvas import CanvasError, TextCanvas
from urwid.display.escape import SAFE_ASCII_DEC_SPECIAL_RE
from urwid.util import apply_target_encoding, str_util
def add_glyphs(self, gdata: str) -> None:
    d, utf8_required = separate_glyphs(gdata, self.height)
    self.char.update(d)
    self.utf8_required |= utf8_required