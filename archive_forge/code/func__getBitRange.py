from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _getBitRange(self, row, bitDepth, metrics):
    rowBits = bitDepth * metrics.width
    bitOffset = row * rowBits
    return (bitOffset, bitOffset + rowBits)