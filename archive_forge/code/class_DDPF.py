from __future__ import annotations
import io
import struct
import sys
from enum import IntEnum, IntFlag
from . import Image, ImageFile, ImagePalette
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o32le as o32
class DDPF(IntFlag):
    ALPHAPIXELS = 1
    ALPHA = 2
    FOURCC = 4
    PALETTEINDEXED8 = 32
    RGB = 64
    LUMINANCE = 131072