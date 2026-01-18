from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
class _PyAccessI16_B(PyAccess):
    """I;16B access, with conversion"""

    def _post_init(self, *args, **kwargs):
        self.pixels = ffi.cast('struct Pixel_I16 **', self.image)

    def get_pixel(self, x, y):
        pixel = self.pixels[y][x]
        return pixel.l * 256 + pixel.r

    def set_pixel(self, x, y, color):
        pixel = self.pixels[y][x]
        try:
            color = min(color, 65535)
        except Exception:
            color = min(color[0], 65535)
        pixel.l = color >> 8
        pixel.r = color & 255