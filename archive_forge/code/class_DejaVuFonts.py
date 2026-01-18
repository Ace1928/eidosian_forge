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
class DejaVuFonts(UnicodeFonts, metaclass=abc.ABCMeta):
    _fontmap: dict[str | int, str] = {}

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        if isinstance(self, DejaVuSerifFonts):
            self._fallback_font = StixFonts(default_font_prop, load_glyph_flags)
        else:
            self._fallback_font = StixSansFonts(default_font_prop, load_glyph_flags)
        self.bakoma = BakomaFonts(default_font_prop, load_glyph_flags)
        TruetypeFonts.__init__(self, default_font_prop, load_glyph_flags)
        self._fontmap.update({1: 'STIXSizeOneSym', 2: 'STIXSizeTwoSym', 3: 'STIXSizeThreeSym', 4: 'STIXSizeFourSym', 5: 'STIXSizeFiveSym'})
        for key, name in self._fontmap.items():
            fullpath = findfont(name)
            self.fontmap[key] = fullpath
            self.fontmap[name] = fullpath

    def _get_glyph(self, fontname: str, font_class: str, sym: str) -> tuple[FT2Font, int, bool]:
        if sym == '\\prime':
            return self.bakoma._get_glyph(fontname, font_class, sym)
        else:
            uniindex = get_unicode_index(sym)
            font = self._get_font('ex')
            if font is not None:
                glyphindex = font.get_char_index(uniindex)
                if glyphindex != 0:
                    return super()._get_glyph('ex', font_class, sym)
            return super()._get_glyph(fontname, font_class, sym)