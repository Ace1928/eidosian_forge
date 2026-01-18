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
def _set_glue(self, x: float, sign: int, totals: list[float], error_type: str) -> None:
    self.glue_order = o = next((i for i in range(len(totals))[::-1] if totals[i] != 0), 0)
    self.glue_sign = sign
    if totals[o] != 0.0:
        self.glue_set = x / totals[o]
    else:
        self.glue_sign = 0
        self.glue_ratio = 0.0
    if o == 0:
        if len(self.children):
            _log.warning('%s %s: %r', error_type, type(self).__name__, self)