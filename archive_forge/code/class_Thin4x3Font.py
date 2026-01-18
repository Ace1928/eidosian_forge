from __future__ import annotations
import typing
import warnings
from pprint import pformat
from urwid.canvas import CanvasError, TextCanvas
from urwid.display.escape import SAFE_ASCII_DEC_SPECIAL_RE
from urwid.util import apply_target_encoding, str_util
class Thin4x3Font(Font):
    name = 'Thin 4x3'
    height = 3
    data = (*Thin3x3Font.data, '\n0000111122223333444455556666777788889999  ####$$$$\n┌──┐  ┐ ┌──┐┌──┐   ┐┌── ┌── ┌──┐┌──┐┌──┐   ┼─┼┌┼┼┐\n│  │  │ ┌──┘  ─┤└──┼└──┐├──┐   ┼├──┤└──┤   ┼─┼└┼┼┐\n└──┘  ┴ └── └──┘   ┴ ──┘└──┘   ┴└──┘ ──┘      └┼┼┘\n')