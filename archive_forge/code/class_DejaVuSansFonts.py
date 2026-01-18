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
class DejaVuSansFonts(DejaVuFonts):
    """
    A font handling class for the DejaVu Sans fonts

    If a glyph is not found it will fallback to Stix Sans
    """
    _fontmap = {'rm': 'DejaVu Sans', 'it': 'DejaVu Sans:italic', 'bf': 'DejaVu Sans:weight=bold', 'bfit': 'DejaVu Sans:italic:bold', 'sf': 'DejaVu Sans', 'tt': 'DejaVu Sans Mono', 'ex': 'DejaVu Sans Display', 0: 'DejaVu Sans'}