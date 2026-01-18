from __future__ import annotations
import io
import struct
import sys
from enum import IntEnum, IntFlag
from . import Image, ImageFile, ImagePalette
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o32le as o32
class DDSD(IntFlag):
    CAPS = 1
    HEIGHT = 2
    WIDTH = 4
    PITCH = 8
    PIXELFORMAT = 4096
    MIPMAPCOUNT = 131072
    LINEARSIZE = 524288
    DEPTH = 8388608