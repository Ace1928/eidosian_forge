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
class VectorParse(NamedTuple):
    """
    The namedtuple type returned by ``MathTextParser("path").parse(...)``.

    Attributes
    ----------
    width, height, depth : float
        The global metrics.
    glyphs : list
        The glyphs including their positions.
    rect : list
        The list of rectangles.
    """
    width: float
    height: float
    depth: float
    glyphs: list[tuple[FT2Font, float, int, float, float]]
    rects: list[tuple[float, float, float, float]]