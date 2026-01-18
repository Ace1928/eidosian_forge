from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
def _accept(prefix):
    return prefix[:4] in (b'BLP1', b'BLP2')