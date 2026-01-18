from __future__ import annotations
import itertools
import logging
import re
import struct
import warnings
import zlib
from enum import IntEnum
from . import Image, ImageChops, ImageFile, ImagePalette, ImageSequence
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from ._binary import o32be as o32
class Blend(IntEnum):
    OP_SOURCE = 0
    '\n    All color components of this frame, including alpha, overwrite the previous output\n    image contents.\n    See :ref:`Saving APNG sequences<apng-saving>`.\n    '
    OP_OVER = 1
    '\n    This frame should be alpha composited with the previous output image contents.\n    See :ref:`Saving APNG sequences<apng-saving>`.\n    '