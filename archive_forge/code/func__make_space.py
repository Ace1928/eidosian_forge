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
def _make_space(self, percentage: float) -> Kern:
    state = self.get_state()
    key = (state.font, state.fontsize, state.dpi)
    width = self._em_width_cache.get(key)
    if width is None:
        metrics = state.fontset.get_metrics('it', mpl.rcParams['mathtext.default'], 'm', state.fontsize, state.dpi)
        width = metrics.advance
        self._em_width_cache[key] = width
    return Kern(width * percentage)