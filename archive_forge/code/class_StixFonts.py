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
class StixFonts(UnicodeFonts):
    """
    A font handling class for the STIX fonts.

    In addition to what UnicodeFonts provides, this class:

    - supports "virtual fonts" which are complete alpha numeric
      character sets with different font styles at special Unicode
      code points, such as "Blackboard".

    - handles sized alternative characters for the STIXSizeX fonts.
    """
    _fontmap: dict[str | int, str] = {'rm': 'STIXGeneral', 'it': 'STIXGeneral:italic', 'bf': 'STIXGeneral:weight=bold', 'bfit': 'STIXGeneral:italic:bold', 'nonunirm': 'STIXNonUnicode', 'nonuniit': 'STIXNonUnicode:italic', 'nonunibf': 'STIXNonUnicode:weight=bold', 0: 'STIXGeneral', 1: 'STIXSizeOneSym', 2: 'STIXSizeTwoSym', 3: 'STIXSizeThreeSym', 4: 'STIXSizeFourSym', 5: 'STIXSizeFiveSym'}
    _fallback_font = None
    _sans = False

    def __init__(self, default_font_prop: FontProperties, load_glyph_flags: int):
        TruetypeFonts.__init__(self, default_font_prop, load_glyph_flags)
        for key, name in self._fontmap.items():
            fullpath = findfont(name)
            self.fontmap[key] = fullpath
            self.fontmap[name] = fullpath

    def _map_virtual_font(self, fontname: str, font_class: str, uniindex: int) -> tuple[str, int]:
        font_mapping = stix_virtual_fonts.get(fontname)
        if self._sans and font_mapping is None and (fontname not in ('regular', 'default')):
            font_mapping = stix_virtual_fonts['sf']
            doing_sans_conversion = True
        else:
            doing_sans_conversion = False
        if isinstance(font_mapping, dict):
            try:
                mapping = font_mapping[font_class]
            except KeyError:
                mapping = font_mapping['rm']
        elif isinstance(font_mapping, list):
            mapping = font_mapping
        else:
            mapping = None
        if mapping is not None:
            lo = 0
            hi = len(mapping)
            while lo < hi:
                mid = (lo + hi) // 2
                range = mapping[mid]
                if uniindex < range[0]:
                    hi = mid
                elif uniindex <= range[1]:
                    break
                else:
                    lo = mid + 1
            if range[0] <= uniindex <= range[1]:
                uniindex = uniindex - range[0] + range[3]
                fontname = range[2]
            elif not doing_sans_conversion:
                uniindex = 1
                fontname = mpl.rcParams['mathtext.default']
        if fontname in ('rm', 'it'):
            uniindex = stix_glyph_fixes.get(uniindex, uniindex)
        if fontname in ('it', 'rm', 'bf', 'bfit') and 57344 <= uniindex <= 63743:
            fontname = 'nonuni' + fontname
        return (fontname, uniindex)

    @functools.cache
    def get_sized_alternatives_for_symbol(self, fontname: str, sym: str) -> list[tuple[str, str]] | list[tuple[int, str]]:
        fixes = {'\\{': '{', '\\}': '}', '\\[': '[', '\\]': ']', '<': '⟨', '>': '⟩'}
        sym = fixes.get(sym, sym)
        try:
            uniindex = get_unicode_index(sym)
        except ValueError:
            return [(fontname, sym)]
        alternatives = [(i, chr(uniindex)) for i in range(6) if self._get_font(i).get_char_index(uniindex) != 0]
        if sym == '\\__sqrt__':
            alternatives = alternatives[:-1]
        return alternatives