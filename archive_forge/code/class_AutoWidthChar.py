from __future__ import annotations
import abc
import copy
import enum
import functools
import logging
import os
import re
import types
import unicodedata
import string
import typing as T
from typing import NamedTuple
import numpy as np
from pyparsing import (
import matplotlib as mpl
from . import cbook
from ._mathtext_data import (
from .font_manager import FontProperties, findfont, get_font
from .ft2font import FT2Font, FT2Image, KERNING_DEFAULT
from packaging.version import parse as parse_version
from pyparsing import __version__ as pyparsing_version
class AutoWidthChar(Hlist):
    """
    A character as close to the given width as possible.

    When using a font with multiple width versions of some characters (such as
    the BaKoMa fonts), the correct glyph will be selected, otherwise this will
    always just return a scaled version of the glyph.
    """

    def __init__(self, c: str, width: float, state: ParserState, always: bool=False, char_class: type[Char]=Char):
        alternatives = state.fontset.get_sized_alternatives_for_symbol(state.font, c)
        state = state.copy()
        for fontname, sym in alternatives:
            state.font = fontname
            char = char_class(sym, state)
            if char.width >= width:
                break
        factor = width / char.width
        state.fontsize *= factor
        char = char_class(sym, state)
        super().__init__([char])
        self.width = char.width