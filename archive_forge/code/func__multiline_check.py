from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def _multiline_check(self, text):
    split_character = '\n' if isinstance(text, str) else b'\n'
    return split_character in text