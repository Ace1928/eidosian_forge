from __future__ import annotations
import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
def _limit_rational(val, max_val):
    inv = abs(val) > 1
    n_d = IFDRational(1 / val if inv else val).limit_rational(max_val)
    return n_d[::-1] if inv else n_d