from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
class ebdt_bitmap_format_5(BitAlignedBitmapMixin, BitmapGlyph):

    def decompile(self):
        self.imageData = self.data

    def compile(self, ttFont):
        return self.imageData