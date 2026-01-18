from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
class _PyAccessI16_N(PyAccess):
    """I;16 access, native bitendian without conversion"""

    def _post_init(self, *args, **kwargs):
        self.pixels = ffi.cast('unsigned short **', self.image)

    def get_pixel(self, x, y):
        return self.pixels[y][x]

    def set_pixel(self, x, y, color):
        try:
            self.pixels[y][x] = min(color, 65535)
        except TypeError:
            self.pixels[y][x] = min(color[0], 65535)