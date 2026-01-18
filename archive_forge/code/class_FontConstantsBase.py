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
class FontConstantsBase:
    """
    A set of constants that controls how certain things, such as sub-
    and superscripts are laid out.  These are all metrics that can't
    be reliably retrieved from the font metrics in the font itself.
    """
    script_space: T.ClassVar[float] = 0.05
    subdrop: T.ClassVar[float] = 0.4
    sup1: T.ClassVar[float] = 0.7
    sub1: T.ClassVar[float] = 0.3
    sub2: T.ClassVar[float] = 0.5
    delta: T.ClassVar[float] = 0.025
    delta_slanted: T.ClassVar[float] = 0.2
    delta_integral: T.ClassVar[float] = 0.1