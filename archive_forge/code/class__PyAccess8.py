from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
class _PyAccess8(PyAccess):
    """1, L, P, 8 bit images stored as uint8"""

    def _post_init(self, *args, **kwargs):
        self.pixels = self.image8

    def get_pixel(self, x, y):
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        try:
            self.pixels[y][x] = min(color, 255)
        except TypeError:
            self.pixels[y][x] = min(color[0], 255)