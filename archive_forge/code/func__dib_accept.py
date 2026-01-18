from __future__ import annotations
import os
from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32
def _dib_accept(prefix):
    return i32(prefix) in [12, 40, 64, 108, 124]