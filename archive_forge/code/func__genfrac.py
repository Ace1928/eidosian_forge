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
def _genfrac(self, ldelim: str, rdelim: str, rule: float | None, style: _MathStyle, num: Hlist, den: Hlist) -> T.Any:
    state = self.get_state()
    thickness = state.get_current_underline_thickness()
    for _ in range(style.value):
        num.shrink()
        den.shrink()
    cnum = HCentered([num])
    cden = HCentered([den])
    width = max(num.width, den.width)
    cnum.hpack(width, 'exactly')
    cden.hpack(width, 'exactly')
    vlist = Vlist([cnum, Vbox(0, thickness * 2.0), Hrule(state, rule), Vbox(0, thickness * 2.0), cden])
    metrics = state.fontset.get_metrics(state.font, mpl.rcParams['mathtext.default'], '=', state.fontsize, state.dpi)
    shift = cden.height - ((metrics.ymax + metrics.ymin) / 2 - thickness * 3.0)
    vlist.shift_amount = shift
    result = [Hlist([vlist, Hbox(thickness * 2.0)])]
    if ldelim or rdelim:
        if ldelim == '':
            ldelim = '.'
        if rdelim == '':
            rdelim = '.'
        return self._auto_sized_delimiter(ldelim, T.cast(list[T.Union[Box, Char, str]], result), rdelim)
    return result