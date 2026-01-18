from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _readRawImageData(bitmapObject, name, attrs, content, ttFont):
    bitmapObject.imageData = readHex(content)