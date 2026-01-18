from __future__ import annotations
import os
from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
def _palette(self, palette, shift):
    i = 0
    for e in range(i16(self.fp.read(2))):
        s = self.fp.read(2)
        i = i + s[0]
        n = s[1]
        if n == 0:
            n = 256
        s = self.fp.read(n * 3)
        for n in range(0, len(s), 3):
            r = s[n] << shift
            g = s[n + 1] << shift
            b = s[n + 2] << shift
            palette[i] = (r, g, b)
            i += 1