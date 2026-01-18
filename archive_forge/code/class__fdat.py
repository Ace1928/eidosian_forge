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
class _fdat:

    def __init__(self, fp, chunk, seq_num):
        self.fp = fp
        self.chunk = chunk
        self.seq_num = seq_num

    def write(self, data):
        self.chunk(self.fp, b'fdAT', o32(self.seq_num), data)
        self.seq_num += 1