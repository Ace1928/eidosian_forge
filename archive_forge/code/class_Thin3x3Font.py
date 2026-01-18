from __future__ import annotations
import typing
import warnings
from pprint import pformat
from urwid.canvas import CanvasError, TextCanvas
from urwid.display.escape import SAFE_ASCII_DEC_SPECIAL_RE
from urwid.util import apply_target_encoding, str_util
class Thin3x3Font(Font):
    name = 'Thin 3x3'
    height = 3
    data = ('\n000111222333444555666777888999  !\n┌─┐ ┐ ┌─┐┌─┐  ┐┌─ ┌─ ┌─┐┌─┐┌─┐  │\n│ │ │ ┌─┘ ─┤└─┼└─┐├─┐  ┼├─┤└─┤  │\n└─┘ ┴ └─ └─┘  ┴ ─┘└─┘  ┴└─┘ ─┘  .\n', '\n"###$$$%%%\'*++,--.///:;==???[[\\\\\\]]^__`\n" ┼┼┌┼┐O /\'         /.. _┌─┐┌ \\   ┐^  `\n  ┼┼└┼┐ /  * ┼  ─  / ., _ ┌┘│  \\  │\n    └┼┘/ O    ,  ./       . └   \\ ┘ ──\n')