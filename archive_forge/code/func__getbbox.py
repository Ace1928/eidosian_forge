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
def _getbbox(base_im, im_frame):
    if _get_palette_bytes(im_frame) != _get_palette_bytes(base_im):
        im_frame = im_frame.convert('RGBA')
        base_im = base_im.convert('RGBA')
    delta = ImageChops.subtract_modulo(im_frame, base_im)
    return (delta, delta.getbbox(alpha_only=False))