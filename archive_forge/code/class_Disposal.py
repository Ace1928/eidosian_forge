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
class Disposal(IntEnum):
    OP_NONE = 0
    '\n    No disposal is done on this frame before rendering the next frame.\n    See :ref:`Saving APNG sequences<apng-saving>`.\n    '
    OP_BACKGROUND = 1
    '\n    This frame’s modified region is cleared to fully transparent black before rendering\n    the next frame.\n    See :ref:`Saving APNG sequences<apng-saving>`.\n    '
    OP_PREVIOUS = 2
    '\n    This frame’s modified region is reverted to the previous frame’s contents before\n    rendering the next frame.\n    See :ref:`Saving APNG sequences<apng-saving>`.\n    '