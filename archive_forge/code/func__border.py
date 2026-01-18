from __future__ import annotations
import functools
import operator
import re
from . import ExifTags, Image, ImagePalette
def _border(border):
    if isinstance(border, tuple):
        if len(border) == 2:
            left, top = right, bottom = border
        elif len(border) == 4:
            left, top, right, bottom = border
    else:
        left = top = right = bottom = border
    return (left, top, right, bottom)