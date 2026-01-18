from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
class _PyAccessI32_Swap(PyAccess):
    """I;32L/B access, with byteswapping conversion"""

    def _post_init(self, *args, **kwargs):
        self.pixels = self.image32

    def reverse(self, i):
        orig = ffi.new('int *', i)
        chars = ffi.cast('unsigned char *', orig)
        chars[0], chars[1], chars[2], chars[3] = (chars[3], chars[2], chars[1], chars[0])
        return ffi.cast('int *', chars)[0]

    def get_pixel(self, x, y):
        return self.reverse(self.pixels[y][x])

    def set_pixel(self, x, y, color):
        self.pixels[y][x] = self.reverse(color)