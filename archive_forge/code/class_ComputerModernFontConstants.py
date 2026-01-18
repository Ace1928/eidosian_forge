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
class ComputerModernFontConstants(FontConstantsBase):
    script_space = 0.075
    subdrop = 0.2
    sup1 = 0.45
    sub1 = 0.2
    sub2 = 0.3
    delta = 0.075
    delta_slanted = 0.3
    delta_integral = 0.3