from __future__ import annotations
import math
import numbers
import struct
from . import Image, ImageColor
def _getfont(self, font_size):
    if font_size is not None:
        from . import ImageFont
        font = ImageFont.load_default(font_size)
    else:
        font = self.getfont()
    return font