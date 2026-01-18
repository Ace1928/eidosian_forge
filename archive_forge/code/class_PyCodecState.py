from __future__ import annotations
import io
import itertools
import struct
import sys
from typing import Any, NamedTuple
from . import Image
from ._deprecate import deprecate
from ._util import is_path
class PyCodecState:

    def __init__(self):
        self.xsize = 0
        self.ysize = 0
        self.xoff = 0
        self.yoff = 0

    def extents(self):
        return (self.xoff, self.yoff, self.xoff + self.xsize, self.yoff + self.ysize)