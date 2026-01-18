from __future__ import annotations
import itertools
import math
import os
import subprocess
from enum import IntEnum
from . import (
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16
def _rgb(color):
    if self._frame_palette:
        if color * 3 + 3 > len(self._frame_palette.palette):
            color = 0
        color = tuple(self._frame_palette.palette[color * 3:color * 3 + 3])
    else:
        color = (color, color, color)
    return color