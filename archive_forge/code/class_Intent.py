from __future__ import annotations
import sys
from enum import IntEnum
from . import Image
class Intent(IntEnum):
    PERCEPTUAL = 0
    RELATIVE_COLORIMETRIC = 1
    SATURATION = 2
    ABSOLUTE_COLORIMETRIC = 3