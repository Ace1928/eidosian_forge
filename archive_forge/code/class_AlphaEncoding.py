from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
class AlphaEncoding(IntEnum):
    DXT1 = 0
    DXT3 = 1
    DXT5 = 7