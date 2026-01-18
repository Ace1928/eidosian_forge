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
def _normalize_mode(im):
    """
    Takes an image (or frame), returns an image in a mode that is appropriate
    for saving in a Gif.

    It may return the original image, or it may return an image converted to
    palette or 'L' mode.

    :param im: Image object
    :returns: Image object
    """
    if im.mode in RAWMODE:
        im.load()
        return im
    if Image.getmodebase(im.mode) == 'RGB':
        im = im.convert('P', palette=Image.Palette.ADAPTIVE)
        if im.palette.mode == 'RGBA':
            for rgba in im.palette.colors:
                if rgba[3] == 0:
                    im.info['transparency'] = im.palette.colors[rgba]
                    break
        return im
    return im.convert('L')