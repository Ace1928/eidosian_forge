from __future__ import annotations
import io
import struct
import sys
from enum import IntEnum, IntFlag
from . import Image, ImageFile, ImagePalette
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o32le as o32
class DDSCAPS2(IntFlag):
    CUBEMAP = 512
    CUBEMAP_POSITIVEX = 1024
    CUBEMAP_NEGATIVEX = 2048
    CUBEMAP_POSITIVEY = 4096
    CUBEMAP_NEGATIVEY = 8192
    CUBEMAP_POSITIVEZ = 16384
    CUBEMAP_NEGATIVEZ = 32768
    VOLUME = 2097152