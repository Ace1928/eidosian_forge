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
class Kern(Node):
    """
    A `Kern` node has a width field to specify a (normally
    negative) amount of spacing. This spacing correction appears in
    horizontal lists between letters like A and V when the font
    designer said that it looks better to move them closer together or
    further apart. A kern node can also appear in a vertical list,
    when its *width* denotes additional spacing in the vertical
    direction.
    """
    height = 0
    depth = 0

    def __init__(self, width: float):
        super().__init__()
        self.width = width

    def __repr__(self) -> str:
        return 'k%.02f' % self.width

    def shrink(self) -> None:
        super().shrink()
        if self.size < NUM_SIZE_LEVELS:
            self.width *= SHRINK_FACTOR