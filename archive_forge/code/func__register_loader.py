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
def _register_loader(idx, size):

    def decorator(func):
        from .TiffTags import TYPES
        if func.__name__.startswith('load_'):
            TYPES[idx] = func.__name__[5:].replace('_', ' ')
        _load_dispatch[idx] = (size, func)
        return func
    return decorator