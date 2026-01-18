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
def accent(self, toks: ParseResults) -> T.Any:
    state = self.get_state()
    thickness = state.get_current_underline_thickness()
    accent = toks['accent']
    sym = toks['sym']
    accent_box: Node
    if accent in self._wide_accents:
        accent_box = AutoWidthChar('\\' + accent, sym.width, state, char_class=Accent)
    else:
        accent_box = Accent(self._accent_map[accent], state)
    if accent == 'mathring':
        accent_box.shrink()
        accent_box.shrink()
    centered = HCentered([Hbox(sym.width / 4.0), accent_box])
    centered.hpack(sym.width, 'exactly')
    return Vlist([centered, Vbox(0.0, thickness * 2.0), Hlist([sym])])