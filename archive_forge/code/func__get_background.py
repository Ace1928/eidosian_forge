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
def _get_background(im, info_background):
    background = 0
    if info_background:
        if isinstance(info_background, tuple):
            try:
                background = im.palette.getcolor(info_background, im)
            except ValueError as e:
                if str(e) not in ('cannot allocate more than 256 colors', 'cannot add non-opaque RGBA color to RGB palette'):
                    raise
        else:
            background = info_background
    return background