from __future__ import annotations
import io
import struct
import sys
from enum import IntEnum, IntFlag
from . import Image, ImageFile, ImagePalette
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o32le as o32
class DDSCAPS(IntFlag):
    COMPLEX = 8
    TEXTURE = 4096
    MIPMAP = 4194304