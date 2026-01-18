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
def _get_optimize(im, info):
    """
    Palette optimization is a potentially expensive operation.

    This function determines if the palette should be optimized using
    some heuristics, then returns the list of palette entries in use.

    :param im: Image object
    :param info: encoderinfo
    :returns: list of indexes of palette entries in use, or None
    """
    if im.mode in ('P', 'L') and info and info.get('optimize'):
        optimise = _FORCE_OPTIMIZE or im.mode == 'L'
        if optimise or im.width * im.height < 512 * 512:
            used_palette_colors = []
            for i, count in enumerate(im.histogram()):
                if count:
                    used_palette_colors.append(i)
            if optimise or max(used_palette_colors) >= len(used_palette_colors):
                return used_palette_colors
            num_palette_colors = len(im.palette.palette) // Image.getmodebands(im.palette.mode)
            current_palette_size = 1 << (num_palette_colors - 1).bit_length()
            if len(used_palette_colors) <= current_palette_size // 2 and current_palette_size > 2:
                return used_palette_colors