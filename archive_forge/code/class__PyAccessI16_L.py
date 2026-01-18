from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
class _PyAccessI16_L(PyAccess):
    """I;16L access, with conversion"""

    def _post_init(self, *args, **kwargs):
        self.pixels = ffi.cast('struct Pixel_I16 **', self.image)

    def get_pixel(self, x, y):
        pixel = self.pixels[y][x]
        return pixel.l + pixel.r * 256

    def set_pixel(self, x, y, color):
        pixel = self.pixels[y][x]
        try:
            color = min(color, 65535)
        except TypeError:
            color = min(color[0], 65535)
        pixel.l = color & 255
        pixel.r = color >> 8